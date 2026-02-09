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
    total = sum(int(d) * w for d, w in zip(digits, weights, strict=True))
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

    return int(digits[10]) == check2


def normalize_cpf(cpf: str | None) -> str | None:
    """
    Normalize CPF to standard format ###.###.###-##.

    Handles both traditional (###.###.###-##) and newer (#########/##) formats.

    Returns None if input doesn't have exactly 11 digits.
    """
    if not cpf:
        return None

    digits = _extract_digits(cpf)
    if len(digits) != 11:
        return None

    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:11]}"


def _is_new_cpf_format(value: str | None) -> bool:
    """Check if value matches the newer CPF format #########/##.

    This format is exclusive to CPF — RG numbers never use it.
    The VLM may misread a digit, so we detect by format, not validation.
    """
    if not value:
        return False
    return bool(re.match(r"^\d{9}/\d{2}$", value.strip()))


def fix_cpf_rg_swap(data: dict) -> dict:
    """
    Detect and fix when VLM swaps CPF and RG/doc_identidade fields.

    The VLM sometimes confuses CPF and RG numbers, especially with
    the newer CPF format (#########/##) which looks like an ID number.

    Works with both RG docs (field "rg") and CNH docs (field "doc_identidade").

    Strategy:
    1. If cpf field is valid → normalize and keep as-is
    2. If cpf is invalid but rg/doc_identidade validates as CPF → swap them
    3. If cpf is invalid and rg/doc_identidade matches new CPF format (#########/##) → swap
       (even if check digits don't validate, since VLM may misread digits)
    4. Normalize CPF to ###.###.###-## format
    """
    cpf_val = data.get("cpf")

    # Detect which field holds the RG number (RG docs use "rg", CNH uses "doc_identidade")
    rg_field = "rg" if "rg" in data else "doc_identidade" if "doc_identidade" in data else None
    rg_val = data.get(rg_field) if rg_field else None

    cpf_valid = is_valid_cpf(cpf_val)

    if not cpf_valid and rg_val and is_valid_cpf(rg_val):
        # RG/doc_identidade field contains a valid CPF — swap them
        data["cpf"] = normalize_cpf(rg_val)
        data[rg_field] = cpf_val
    elif not cpf_valid and rg_val and _is_new_cpf_format(rg_val):
        # RG field has new CPF format (#########/##) — definitely swapped
        # Even if check digits don't match (VLM misread), swap by format
        data["cpf"] = normalize_cpf(rg_val)
        data[rg_field] = cpf_val
    elif cpf_valid:
        # Normalize to standard format
        data["cpf"] = normalize_cpf(cpf_val)

    return data


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
