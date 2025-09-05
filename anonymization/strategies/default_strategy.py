"""
Default Replacement Strategy

Fallback strategy for entity types that don't have specialized handlers.
Provides generic placeholder-based replacement.

Author: G Rohit
Version: 1.0.0
"""

import random
import string
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DefaultReplacementStrategy:
    """Default fallback replacement strategy."""

    def __init__(self):
        """Initialize default strategy."""
        self.replacement_patterns = {
            'email': self._generate_email,
            'phone': self._generate_phone,
            'ssn': self._generate_ssn,
            'id': self._generate_id,
            'number': self._generate_number,
            'code': self._generate_code,
            'username': self._generate_username
        }

    def can_handle(self, label: str) -> bool:
        """This strategy can handle any label as fallback."""
        return True

    def get_replacement(self, entity: Dict[str, Any]) -> str:
        """
        Generate a generic replacement for any entity.

        Args:
            entity: Entity dictionary with label, text, etc.

        Returns:
            Generic replacement string
        """
        label = entity['label'].lower()
        original_text = entity['text']

        try:
            # Try specialized pattern if available
            if label in self.replacement_patterns:
                return self.replacement_patterns[label]()

            # Generic replacement based on original text characteristics
            return self._generate_generic_replacement(original_text, label)

        except Exception as e:
            logger.debug(f"Default strategy failed for {label}: {str(e)}")
            return f"[{label.upper()}_REDACTED]"

    def _generate_generic_replacement(self, original_text: str, label: str) -> str:
        """Generate generic replacement based on original text characteristics."""
        text_len = len(original_text)

        # Preserve general structure
        if original_text.isdigit():
            # Replace numbers with random numbers
            return ''.join(random.choices(string.digits, k=text_len))
        elif original_text.isalpha():
            # Replace letters with random letters, preserving case pattern
            result = []
            for char in original_text:
                if char.isupper():
                    result.append(random.choice(string.ascii_uppercase))
                elif char.islower():
                    result.append(random.choice(string.ascii_lowercase))
                else:
                    result.append(char)
            return ''.join(result)
        elif any(char.isalnum() for char in original_text):
            # Mixed alphanumeric - replace alphanumeric chars
            result = []
            for char in original_text:
                if char.isdigit():
                    result.append(random.choice(string.digits))
                elif char.isalpha():
                    if char.isupper():
                        result.append(random.choice(string.ascii_uppercase))
                    else:
                        result.append(random.choice(string.ascii_lowercase))
                else:
                    result.append(char)  # Keep special chars
            return ''.join(result)
        else:
            # Fallback to labeled placeholder
            return f"[{label.upper()}_REDACTED]"

    def _generate_email(self) -> str:
        """Generate a fake email address."""
        domains = ['example.com', 'test.org', 'sample.net', 'demo.co']
        username = ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
        domain = random.choice(domains)
        return f"{username}@{domain}"

    def _generate_phone(self) -> str:
        """Generate a fake phone number."""
        # US format
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)  
        number = random.randint(1000, 9999)
        return f"({area_code}) {exchange}-{number}"

    def _generate_ssn(self) -> str:
        """Generate a fake SSN-like number."""
        return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"

    def _generate_id(self) -> str:
        """Generate a generic ID."""
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        numbers = ''.join(random.choices(string.digits, k=6))
        return f"{letters}{numbers}"

    def _generate_number(self) -> str:
        """Generate a random number."""
        return str(random.randint(100000, 999999))

    def _generate_code(self) -> str:
        """Generate a random code."""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    def _generate_username(self) -> str:
        """Generate a random username.""" 
        prefixes = ['user', 'demo', 'test', 'sample']
        prefix = random.choice(prefixes)
        suffix = random.randint(100, 9999)
        return f"{prefix}{suffix}"
