"""Tests for the GSMArena scraper and its normalisation helpers."""

from __future__ import annotations

import pytest

from src.scraper.gsmarena import GSMArenaScraper
from src.scraper.parsers import (
    clean_text,
    parse_battery_capacity,
    parse_camera_megapixels,
    parse_charging_watts,
    parse_display_size,
    parse_endurance,
    parse_memory_options,
    parse_prices,
    parse_refresh_rate,
    parse_release_year,
    parse_weight,
    slugify,
)
from src.scraper.targets import TARGET_MODELS


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
class TestParsers:
    def test_display_size(self):
        assert parse_display_size("6.8 inches, 114.7 cm2 (~90.2%)") == 6.8
        assert parse_display_size("no size here") is None
        assert parse_display_size(None) is None

    def test_refresh_rate(self):
        assert parse_refresh_rate("Dynamic AMOLED 2X, 120Hz, HDR10+") == 120
        assert parse_refresh_rate("Super AMOLED, 90Hz") == 90
        assert parse_refresh_rate("Super AMOLED") is None

    def test_weight(self):
        assert parse_weight("168 g (5.93 oz)") == 168.0
        assert parse_weight("unknown") is None

    def test_battery_capacity(self):
        assert parse_battery_capacity("Li-Ion 5000 mAh, non-removable") == 5000
        assert parse_battery_capacity("Li-Ion 5,000 mAh") == 5000
        assert parse_battery_capacity("Li-Ion") is None

    def test_charging_watts_returns_the_fastest_figure(self):
        text = "45W wired, 65% in 30 min 15W wireless 4.5W reverse wireless"
        assert parse_charging_watts(text) == 45

    def test_memory_options_separates_ram_from_storage(self):
        ram, storage = parse_memory_options("256GB 12GB RAM, 512GB 12GB RAM, 1TB 12GB RAM")
        assert (ram, storage) == (12, 1024)

    def test_memory_options_handles_missing_values(self):
        assert parse_memory_options(None) == (None, None)
        assert parse_memory_options("") == (None, None)

    def test_camera_megapixels_picks_the_main_sensor(self):
        summary = "200 MP, f/1.7 (wide) | 10 MP, f/2.4 (telephoto) | 12 MP (ultrawide)"
        assert parse_camera_megapixels(summary) == 200.0

    def test_release_year(self):
        assert parse_release_year("2023, February 01") == 2023
        assert parse_release_year(None, "Available. Released 2022, March 11") == 2022
        assert parse_release_year("unknown") is None

    def test_endurance_supports_both_gsmarena_metrics(self):
        assert parse_endurance("Active use score 13:24h") == "Active use score 13:24h"
        assert parse_endurance("Endurance rating 108h") == "Endurance rating 108h"

    def test_prices_parses_every_currency(self):
        parsed = parse_prices("$ 252.84 / € 299.00 / £ 229.99 / ₹ 54,999")
        currencies = {currency for currency, _amount, _raw in parsed}
        assert currencies == {"USD", "EUR", "GBP", "INR"}
        amounts = dict((currency, amount) for currency, amount, _raw in parsed)
        assert amounts["USD"] == 252.84
        assert amounts["INR"] == 54999.0

    def test_prices_on_empty_input(self):
        assert parse_prices(None) == []
        assert parse_prices("About 300 EUR") == []

    def test_clean_text_normalises_bullets_and_whitespace(self):
        assert clean_text("● Nano-SIM  +  eSIM ● Dual SIM") == (
            "Nano-SIM + eSIM | Dual SIM"
        )
        assert clean_text("   ") is None
        assert clean_text(None) is None

    def test_slugify(self):
        assert slugify("Galaxy S23+") == "galaxy-s23-plus"
        assert slugify("Samsung Galaxy Z Fold5") == "samsung-galaxy-z-fold5"


# ---------------------------------------------------------------------------
# Page parsing (offline, against a committed fixture)
# ---------------------------------------------------------------------------
class TestPhonePageParsing:
    @pytest.fixture(scope="class")
    def phone(self, request):
        html = request.getfixturevalue("galaxy_s23_html")
        scraper = GSMArenaScraper.__new__(GSMArenaScraper)  # no HTTP client needed
        return scraper.parse_phone_page(
            html, "https://www.gsmarena.com/samsung_galaxy_s23-12082.php"
        )

    def test_identity(self, phone):
        assert phone.name == "Samsung Galaxy S23"
        assert phone.slug == "samsung-galaxy-s23"
        assert phone.image_url and phone.image_url.startswith("https://")

    def test_display_fields(self, phone):
        assert phone.display_size_inches == 6.1
        assert phone.refresh_rate_hz == 120
        assert "AMOLED" in phone.display_type
        assert "1080 x 2340" in phone.display_resolution

    def test_platform_fields(self, phone):
        assert "Snapdragon 8 Gen 2" in phone.chipset
        assert phone.gpu == "Adreno 740"
        assert phone.cpu and "Octa-core" in phone.cpu

    def test_memory_fields(self, phone):
        assert phone.max_ram_gb == 8
        assert phone.max_storage_gb == 512

    def test_camera_fields(self, phone):
        assert phone.main_camera_mp == 50.0
        assert phone.selfie_camera_mp == 12.0
        assert phone.main_camera_video and "8K" in phone.main_camera_video

    def test_battery_fields(self, phone):
        assert phone.battery_capacity_mah == 3900
        assert phone.charging_watts == 25
        assert phone.battery_endurance is not None

    def test_launch_fields(self, phone):
        assert phone.release_year == 2023
        assert phone.announced and "2023" in phone.announced

    def test_every_spec_row_is_captured(self, phone):
        assert phone.spec_count > 40
        categories = {category for category, _k, _v, _p in phone.specs}
        assert {"Display", "Platform", "Battery", "Memory"} <= categories

    def test_spec_rows_are_well_formed(self, phone):
        for category, key, value, position in phone.specs:
            assert category and key and value
            assert isinstance(position, int)

    def test_prices_extracted(self, phone):
        assert phone.prices
        assert all(amount > 0 for _currency, amount, _raw in phone.prices)


# ---------------------------------------------------------------------------
# Listing discovery (offline)
# ---------------------------------------------------------------------------
class TestListingParsing:
    def test_listing_page_yields_model_urls(self, samsung_listing_html):
        scraper = GSMArenaScraper.__new__(GSMArenaScraper)
        scraper.base_url = "https://www.gsmarena.com/"
        found = scraper._parse_listing_page(samsung_listing_html)

        assert len(found) > 20
        for name, url in found.items():
            assert name
            assert url.startswith("https://www.gsmarena.com/")
            assert url.endswith(".php")


class TestTargets:
    def test_target_list_is_within_the_required_range(self):
        assert 10 <= len(TARGET_MODELS) <= 15

    def test_target_names_are_unique(self):
        assert len(set(TARGET_MODELS)) == len(TARGET_MODELS)
