"""Tests for the extended ANARCI insertion code alphabet."""

from ANARCI import schemes


def test_alphabet_starts_with_single_letters():
    """Test that alphabet starts with A-Z."""
    assert schemes.alphabet[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_alphabet_has_double_letters():
    """Test that alphabet includes double-letter codes."""
    # After 26 single letters, should have AA, AB, AC, ...
    assert "AA" in schemes.alphabet
    assert "AB" in schemes.alphabet
    assert "ZZ" in schemes.alphabet


def test_alphabet_ends_with_blank():
    """Test that alphabet ends with blank space for no insertion."""
    assert schemes.alphabet[-1] == " "


def test_alphabet_has_sufficient_capacity():
    """Test that alphabet has at least 1000 codes (default)."""
    # Default max_codes is 1000
    # 26 single + 676 double = 702, so we need some triple letters
    assert len(schemes.alphabet) >= 700


def test_generate_extended_alphabet_custom_size():
    """Test generating alphabet with custom size."""
    small_alphabet = schemes._generate_extended_alphabet(max_codes=50)
    # Should have 26 single letters + 24 double letters + blank = 51
    assert len(small_alphabet) == 51
    assert small_alphabet[-1] == " "


def test_generate_extended_alphabet_includes_triple():
    """Test that very large alphabets include triple-letter codes."""
    large_alphabet = schemes._generate_extended_alphabet(max_codes=800)
    # 26 single + 676 double = 702
    # So we need 98 more from triple letters
    assert "AAA" in large_alphabet
    assert "AAB" in large_alphabet


def test_alphabet_no_duplicates():
    """Test that alphabet has no duplicate entries (except the final blank)."""
    codes_without_blank = schemes.alphabet[:-1]
    assert len(codes_without_blank) == len(set(codes_without_blank))


def test_alphabet_order():
    """Test that alphabet is ordered correctly: A-Z, then AA-AZ, BA-BZ, etc."""
    # Check single letters come first
    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        assert schemes.alphabet[i] == letter

    # Check first double letters come after single letters
    assert schemes.alphabet[26] == "AA"
    assert schemes.alphabet[27] == "AB"
    # AZ should be at index 26 + 25 = 51
    assert schemes.alphabet[51] == "AZ"
    # BA should be at index 26 + 26 = 52
    assert schemes.alphabet[52] == "BA"
