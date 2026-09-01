"""Tests for the database schema and the repository query layer."""

from __future__ import annotations

import pytest

from src.database.models import Base, Phone, Price, SpecCategory, Specification


class TestSchema:
    def test_expected_tables_are_declared(self):
        assert set(Base.metadata.tables) == {
            "phones",
            "spec_categories",
            "specifications",
            "prices",
        }

    def test_relationships_are_wired(self):
        assert Phone.specifications.property.mapper.class_ is Specification
        assert Phone.prices.property.mapper.class_ is Price
        assert Specification.category.property.mapper.class_ is SpecCategory


class TestRepositoryReads:
    def test_all_target_phones_are_stored(self, populated_repository):
        assert populated_repository.count_phones() >= 10

    def test_specifications_are_stored_for_every_phone(self, populated_repository):
        for phone in populated_repository.list_phones():
            assert phone.specifications, f"{phone.name} has no specifications"

    def test_headline_columns_are_populated(self, populated_repository):
        for phone in populated_repository.list_phones():
            assert phone.display_size_inches is not None, phone.name
            assert phone.battery_capacity_mah is not None, phone.name
            assert phone.chipset, phone.name
            assert phone.release_year is not None, phone.name

    def test_find_by_name_is_fuzzy(self, populated_repository):
        phone = populated_repository.find_by_name("s23 ultra")
        assert phone is not None
        assert "S23 Ultra" in phone.name

    def test_find_by_name_returns_none_for_unknown_model(self, populated_repository):
        assert populated_repository.find_by_name("Nokia 3310") is None
        assert populated_repository.find_by_name("   ") is None

    def test_search_by_name(self, populated_repository):
        results = populated_repository.search_by_name("Galaxy S2")
        assert len(results) >= 5

    def test_top_by_column_ranks_correctly(self, populated_repository):
        ranked = populated_repository.top_by_column("battery_capacity_mah", limit=5)
        capacities = [p.battery_capacity_mah for p in ranked]
        assert capacities == sorted(capacities, reverse=True)

    def test_top_by_column_rejects_unknown_column(self, populated_repository):
        with pytest.raises(ValueError):
            populated_repository.top_by_column("does_not_exist")

    def test_specs_are_grouped_by_category(self, populated_repository):
        phone = populated_repository.list_phones(limit=1)[0]
        grouped = populated_repository.specs_by_category(phone)
        assert "Display" in grouped
        assert all(isinstance(v, list) and v for v in grouped.values())

    def test_statistics(self, populated_repository):
        stats = populated_repository.statistics()
        assert stats["phones"] >= 10
        assert stats["specifications"] > 100
        assert stats["release_years"]


class TestDataIntegrity:
    def test_slugs_are_unique(self, populated_repository):
        phones = populated_repository.list_phones()
        slugs = [p.slug for p in phones]
        assert len(set(slugs)) == len(slugs)

    def test_prices_are_positive(self, populated_repository):
        for phone in populated_repository.list_phones():
            for price in phone.prices:
                assert float(price.amount) > 0
                assert len(price.currency) == 3

    def test_numeric_ranges_are_plausible(self, populated_repository):
        for phone in populated_repository.list_phones():
            assert 4.0 < float(phone.display_size_inches) < 9.0, phone.name
            assert 1500 < phone.battery_capacity_mah < 12000, phone.name
            if phone.max_ram_gb:
                assert 2 <= phone.max_ram_gb <= 32, phone.name
            if phone.weight_grams:
                assert 100 < float(phone.weight_grams) < 400, phone.name
