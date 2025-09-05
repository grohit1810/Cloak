"""
Country Replacement Strategy

Specialized strategy for handling country and location entities.
Maintains geographical context while protecting sensitive location data.

Author: G Rohit  
Version: 1.0.0
"""

import random
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class CountryReplacementStrategy:
    """Replacement strategy for countries and locations."""

    def __init__(self):
        """Initialize with country data."""
        self.countries = self._load_countries()
        self.supported_labels = {'country', 'location', 'nationality', 'place'}

    def _load_countries(self) -> List[str]:
        """Load country list from data file or use default list."""
        try:
            # Try to load from data file
            data_file = Path(__file__).parent.parent.parent / 'data' / 'countries.json'
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('countries', self._get_default_countries())
        except Exception as e:
            logger.debug(f"Could not load countries data file: {e}")

        return self._get_default_countries()

    def _get_default_countries(self) -> List[str]:
        """Get default list of countries."""
        return [
            'United States', 'Canada', 'United Kingdom', 'Germany', 'France',
            'Italy', 'Spain', 'Netherlands', 'Belgium', 'Switzerland',
            'Austria', 'Sweden', 'Norway', 'Denmark', 'Finland',
            'Australia', 'New Zealand', 'Japan', 'South Korea', 'Singapore',
            'Brazil', 'Argentina', 'Mexico', 'Chile', 'Colombia',
            'India', 'China', 'Thailand', 'Vietnam', 'Philippines',
            'South Africa', 'Egypt', 'Morocco', 'Kenya', 'Ghana',
            'Russia', 'Poland', 'Czech Republic', 'Hungary', 'Romania'
        ]

    def can_handle(self, label: str) -> bool:
        """Check if this strategy can handle the given label."""
        return label.lower() in self.supported_labels

    def get_replacement(self, entity: Dict[str, Any]) -> Optional[str]:
        """
        Generate a country/location replacement.

        Args:
            entity: Entity dictionary with label, text, etc.

        Returns:
            Country name or None if failed
        """
        original_text = entity['text'].strip()
        label = entity['label'].lower()

        try:
            # Check if the original text looks like a country
            if self._is_country_like(original_text):
                # Replace with a different country
                available_countries = [c for c in self.countries if c.lower() != original_text.lower()]
                if available_countries:
                    return random.choice(available_countries)
            elif label == 'location':
                # For generic locations, return a country
                return random.choice(self.countries)
            elif label == 'nationality':
                # Convert country to nationality (simplified)
                country = random.choice(self.countries)
                return self._country_to_nationality(country)

        except Exception as e:
            logger.debug(f"Country strategy failed for {original_text}: {str(e)}")

        return None

    def _is_country_like(self, text: str) -> bool:
        """Check if text looks like a country name."""
        text_lower = text.lower()

        # Check against known countries (case-insensitive)
        for country in self.countries:
            if country.lower() == text_lower:
                return True

        # Check for common country indicators
        country_indicators = [
            'united states', 'usa', 'uk', 'united kingdom',
            'china', 'india', 'brazil', 'russia', 'japan'
        ]

        return any(indicator in text_lower for indicator in country_indicators)

    def _country_to_nationality(self, country: str) -> str:
        """Convert country name to nationality (simplified)."""
        # Simplified mapping - in production, use a comprehensive mapping
        nationality_mapping = {
            'United States': 'American',
            'United Kingdom': 'British', 
            'Germany': 'German',
            'France': 'French',
            'Italy': 'Italian',
            'Spain': 'Spanish',
            'Canada': 'Canadian',
            'Australia': 'Australian',
            'Japan': 'Japanese',
            'China': 'Chinese',
            'India': 'Indian',
            'Brazil': 'Brazilian'
        }

        return nationality_mapping.get(country, f"{country}n")
