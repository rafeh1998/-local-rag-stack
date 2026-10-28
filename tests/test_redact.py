from app.redact import redact


def test_email_is_redacted():
    r = redact("Contact max.mustermann@example.de for details.")
    assert "example.de" not in r.text
    assert "[EMAIL]" in r.text
    assert r.counts == {"EMAIL": 1}


def test_german_phone_numbers_are_redacted():
    r = redact("Call +49 69 1234 5678 or 0151/23456789.")
    assert r.counts == {"PHONE": 2}
    assert "5678" not in r.text


def test_iban_is_redacted_and_not_counted_as_phone():
    r = redact("Pay to DE89 3704 0044 0532 0130 00 by Friday.")
    assert r.counts == {"IBAN": 1}
    assert "3704" not in r.text


def test_normal_text_and_dates_are_kept():
    text = "Revenue grew 12% in 2025. The meeting is on 14.10.2026 at 10:00."
    r = redact(text)
    assert r.text == text
    assert r.total == 0


def test_amounts_and_years_are_not_phone_numbers():
    text = "Budget: 1.234.567 EUR for 2026, invoice no. 2026-10-14."
    assert redact(text).total == 0
