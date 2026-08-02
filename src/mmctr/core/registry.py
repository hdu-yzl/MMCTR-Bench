"""Dependency-light registry primitives with lazy implementation loading."""

import importlib
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


_CANONICAL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class RegistryError(ValueError):
    """Raised for invalid, duplicate, or unknown registry entries."""


@dataclass(frozen=True)
class ComponentSpec:
    """Lazy import information and declarative component capabilities."""

    name: str
    module: str
    symbol: str
    aliases: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.lower()
        aliases = tuple(alias.lower() for alias in self.aliases)
        for value in (name,) + aliases:
            if not _CANONICAL_NAME.match(value):
                raise RegistryError("registry names must use snake_case: {!r}".format(value))
        if not self.module or not self.symbol:
            raise RegistryError("component module and symbol are required")
        if len(set(aliases)) != len(aliases) or name in aliases:
            raise RegistryError("component aliases must be unique and differ from the name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def load(self) -> Any:
        module = importlib.import_module(self.module)
        try:
            return getattr(module, self.symbol)
        except AttributeError as error:
            raise RegistryError(
                "component {!r} does not exist in module {!r}".format(self.symbol, self.module)
            ) from error


class ComponentRegistry:
    """Name-to-spec registry that does not import implementations until resolution."""

    def __init__(self, kind: str) -> None:
        if not kind:
            raise RegistryError("registry kind is required")
        self.kind = kind
        self._specs: Dict[str, ComponentSpec] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, spec: ComponentSpec) -> None:
        claimed = (spec.name,) + spec.aliases
        collisions = [
            name for name in claimed if name in self._specs or name in self._aliases
        ]
        if collisions:
            raise RegistryError(
                "duplicate {} registry names: {}".format(self.kind, sorted(collisions))
            )
        self._specs[spec.name] = spec
        for alias in spec.aliases:
            self._aliases[alias] = spec.name

    def register_many(self, specs: Iterable[ComponentSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def canonical_name(self, name: str) -> str:
        value = str(name).lower()
        canonical = self._aliases.get(value, value)
        if canonical not in self._specs:
            raise RegistryError("unknown {}: {!r}".format(self.kind, name))
        return canonical

    def spec(self, name: str) -> ComponentSpec:
        return self._specs[self.canonical_name(name)]

    def resolve(self, name: str) -> Any:
        return self.spec(name).load()

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.resolve(name)(*args, **kwargs)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._specs))

    def aliases(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._aliases))


__all__ = ["ComponentRegistry", "ComponentSpec", "RegistryError"]
