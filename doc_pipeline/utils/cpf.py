"""
CPF validation utilities.

CPF (Cadastro de Pessoas Físicas) is the Brazilian individual taxpayer registry.
It has 11 digits where the last 2 are check digits calculated from the first 9.
"""

import re


def _extract_digits(cpf: str) -> str:
    """Extract only digits from CPF string."""
    return re.sub(r"\D", "", cpf)


def _calculate_check_digit(digits: str, weights: list[int]) -> int:
    """Calculate a CPF check digit."""
    total = sum(int(d) * w for d, w in zip(digits, weights))
    remainder = total % 11
    return 0 if remainder < 2 else 11 - remainder


def is_valid_cpf(cpf: str | None) -> bool:
    """
    Validate a CPF number.

    Args:
        cpf: CPF string (with or without formatting)

    Returns:
        True if valid, False otherwise
    """
    if not cpf:
        return False

    # Extract only digits
    digits = _extract_digits(cpf)

    # Must have exactly 11 digits
    if len(digits) != 11:
        return False

    # Check for known invalid CPFs (all same digit)
    if digits == digits[0] * 11:
        return False

    # Calculate first check digit (10th digit)
    # Weights: 10, 9, 8, 7, 6, 5, 4, 3, 2 for first 9 digits
    weights1 = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    check1 = _calculate_check_digit(digits[:9], weights1)

    if int(digits[9]) != check1:
        return False

    # Calculate second check digit (11th digit)
    # Weights: 11, 10, 9, 8, 7, 6, 5, 4, 3, 2 for first 10 digits
    weights2 = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    check2 = _calculate_check_digit(digits[:10], weights2)

    if int(digits[10]) != check2:
        return False

    return True


def validate_cpf(cpf: str | None) -> dict:
    """
    Validate a CPF and return detailed result.

    Args:
        cpf: CPF string (with or without formatting)

    Returns:
        Dict with validation result:
        - valid: bool - whether CPF is valid
        - cpf: str | None - the CPF that was validated
        - error: str | None - error description if invalid
    """
    if not cpf:
        return {
            "valid": False,
            "cpf": None,
            "error": "CPF não informado",
        }

    digits = _extract_digits(cpf)

    if len(digits) != 11:
        return {
            "valid": False,
            "cpf": cpf,
            "error": f"CPF deve ter 11 dígitos, encontrado {len(digits)}",
        }

    if digits == digits[0] * 11:
        return {
            "valid": False,
            "cpf": cpf,
            "error": "CPF inválido (todos os dígitos iguais)",
        }

    # Calculate check digits
    weights1 = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    check1 = _calculate_check_digit(digits[:9], weights1)

    weights2 = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    check2 = _calculate_check_digit(digits[:10], weights2)

    expected_check = f"{check1}{check2}"
    actual_check = digits[9:11]

    if expected_check != actual_check:
        return {
            "valid": False,
            "cpf": cpf,
            "error": f"Dígitos verificadores inválidos (esperado {expected_check}, encontrado {actual_check})",
        }

    return {
        "valid": True,
        "cpf": cpf,
        "error": None,
    }
