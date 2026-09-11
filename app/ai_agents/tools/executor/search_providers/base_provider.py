from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSearchProvider(ABC):
    """
    Abstract base class for all search providers.
    """

    @abstractmethod
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a search query and return a list of results.

        The output list must enforce this dictionary format:
        {"title": str, "url": str, "content": str}
        """
        pass
