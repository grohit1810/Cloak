"""
Date Replacement Strategy  

Specialized strategy for handling date entities while preserving format context.
Supports various date formats and maintains temporal plausibility.

Author: G Rohit
Version: 1.0.0  
"""

import re
import random
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DateReplacementStrategy:
    """Replacement strategy for dates with format preservation."""

    def __init__(self, faker_instance=None):
        """
        Initialize date strategy.

        Args:
            faker_instance: Optional Faker instance for date generation
        """
        self.faker = faker_instance
        self.supported_labels = {
            'date', 'time', 'datetime', 'day', 'month', 'year',
            'birthday', 'dob', 'date_of_birth'
        }

        # Date pattern matching
        self.date_patterns = {
            r'\b\d{1,2}/\d{1,2}/\d{4}\b': self._replace_mdy_format,           # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b': self._replace_mdy_dash_format,      # MM-DD-YYYY  
            r'\b\d{4}-\d{1,2}-\d{1,2}\b': self._replace_ymd_format,           # YYYY-MM-DD
            r'\b\d{1,2}\s+\w+\s+\d{4}\b': self._replace_text_format,        # DD Month YYYY
            r'\b\w+\s+\d{1,2},?\s+\d{4}\b': self._replace_month_text_format, # Month DD, YYYY
            r'\b\d{4}\b': self._replace_year_only,                              # YYYY only
        }

    def can_handle(self, label: str) -> bool:
        """Check if this strategy can handle the given label."""
        return label.lower() in self.supported_labels

    def get_replacement(self, entity: Dict[str, Any]) -> Optional[str]:
        """
        Generate a date replacement preserving format.

        Args:
            entity: Entity dictionary with label, text, etc.

        Returns:
            Generated date string or None if failed
        """
        original_text = entity['text'].strip()
        label = entity['label'].lower()

        try:
            # Try pattern-based replacement first
            for pattern, replacement_func in self.date_patterns.items():
                if re.match(pattern, original_text, re.IGNORECASE):
                    result = replacement_func(original_text)
                    if result:
                        return result

            # Try Faker-based replacement
            if self.faker:
                return self._faker_date_replacement(label)

            # Fallback to simple replacement
            return self._simple_date_replacement(original_text)

        except Exception as e:
            logger.debug(f"Date strategy failed for '{original_text}': {str(e)}")
            return None

    def _replace_mdy_format(self, original: str) -> str:
        """Replace MM/DD/YYYY format."""
        if self.faker:
            fake_date = self.faker.date_between(start_date='-30y', end_date='today')
            return fake_date.strftime('%m/%d/%Y')
        return self._random_mdy_date()

    def _replace_mdy_dash_format(self, original: str) -> str:
        """Replace MM-DD-YYYY format."""
        if self.faker:
            fake_date = self.faker.date_between(start_date='-30y', end_date='today')  
            return fake_date.strftime('%m-%d-%Y')
        return self._random_mdy_date().replace('/', '-')

    def _replace_ymd_format(self, original: str) -> str:
        """Replace YYYY-MM-DD format."""
        if self.faker:
            fake_date = self.faker.date_between(start_date='-30y', end_date='today')
            return fake_date.strftime('%Y-%m-%d')
        return f"{random.randint(1990, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

    def _replace_text_format(self, original: str) -> str:
        """Replace 'DD Month YYYY' format."""
        months = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        if self.faker:
            fake_date = self.faker.date_between(start_date='-30y', end_date='today')
            return fake_date.strftime('%d %B %Y')

        day = random.randint(1, 28)
        month = random.choice(months)
        year = random.randint(1990, 2023)
        return f"{day} {month} {year}"

    def _replace_month_text_format(self, original: str) -> str:
        """Replace 'Month DD, YYYY' format.""" 
        months = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        if self.faker:
            fake_date = self.faker.date_between(start_date='-30y', end_date='today')
            return fake_date.strftime('%B %d, %Y')

        month = random.choice(months)
        day = random.randint(1, 28)
        year = random.randint(1990, 2023)
        return f"{month} {day}, {year}"

    def _replace_year_only(self, original: str) -> str:
        """Replace standalone year."""
        if self.faker:
            return str(self.faker.year())
        return str(random.randint(1990, 2023))

    def _faker_date_replacement(self, label: str) -> str:
        """Generate date using Faker based on label."""
        if label in ['birthday', 'dob', 'date_of_birth']:
            # Generate older dates for birthdays
            fake_date = self.faker.date_between(start_date='-80y', end_date='-18y')
        else:
            # General date range
            fake_date = self.faker.date_between(start_date='-10y', end_date='today')

        return fake_date.strftime('%Y-%m-%d')

    def _simple_date_replacement(self, original: str) -> str:
        """Simple fallback date replacement."""
        year = random.randint(2000, 2023)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Safe day range
        return f"{year}-{month:02d}-{day:02d}"

    def _random_mdy_date(self) -> str:
        """Generate random MM/DD/YYYY date."""
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        year = random.randint(1990, 2023)
        return f"{month:02d}/{day:02d}/{year}"
