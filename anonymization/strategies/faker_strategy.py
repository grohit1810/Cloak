"""
Faker Replacement Strategy

Provides realistic synthetic data generation using the Faker library.
Handles common entity types with appropriate Faker providers.

Author: G Rohit
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FakerReplacementStrategy:
    """Replacement strategy using Faker for realistic synthetic data."""

    def __init__(self, faker_instance):
        """
        Initialize with Faker instance.

        Args:
            faker_instance: Faker instance to use for generation
        """
        self.faker = faker_instance
        self.supported_labels = {
            'person', 'name', 'first_name', 'last_name',
            'email', 'phone', 'address', 'company', 'organization',
            'city', 'state', 'country', 'age', 'job', 'profession'
        }

        # Label to Faker method mapping
        self.method_mapping = {
            'person': 'name',
            'name': 'name',
            'first_name': 'first_name',
            'last_name': 'last_name',
            'email': 'email',
            'phone': 'phone_number',
            'address': 'address',
            'company': 'company',
            'organization': 'company',
            'city': 'city',
            'state': 'state',
            'age': lambda: str(self.faker.random_int(min=18, max=80)),
            'job': 'job',
            'profession': 'job'
        }

    def can_handle(self, label: str) -> bool:
        """Check if this strategy can handle the given label."""
        return self.faker is not None and label.lower() in self.supported_labels

    def get_replacement(self, entity: Dict[str, Any]) -> Optional[str]:
        """
        Generate a replacement using Faker.

        Args:
            entity: Entity dictionary with label, text, etc.

        Returns:
            Generated replacement string or None if failed
        """
        if not self.faker:
            return None

        label = entity['label'].lower()
        original_text = entity['text']

        try:
            # Get the appropriate Faker method
            if label in self.method_mapping:
                method = self.method_mapping[label]

                if callable(method):
                    # Custom lambda function
                    replacement = method()
                else:
                    # Faker method name
                    if hasattr(self.faker, method):
                        faker_method = getattr(self.faker, method)
                        replacement = faker_method()
                    else:
                        logger.debug(f"Faker method '{method}' not available")
                        return None

                # Ensure we don't return the same value
                if replacement == original_text:
                    # Try again with a different generation
                    for _ in range(3):  # Max 3 attempts
                        if callable(method):
                            new_replacement = method()
                        else:
                            faker_method = getattr(self.faker, method)
                            new_replacement = faker_method()
                        if new_replacement != original_text:
                            replacement = new_replacement
                            break

                return replacement

        except Exception as e:
            logger.debug(f"Faker strategy failed for {label}: {str(e)}")
            return None

        return None
