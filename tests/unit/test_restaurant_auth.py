"""Unit tests for per-restaurant credential generation/verification."""

from src.utils.restaurant_auth import (
    generate_restaurant_key,
    hash_restaurant_key,
    verify_restaurant_key,
)


class TestGenerateRestaurantKey:
    def test_returns_a_nonempty_string(self) -> None:
        key = generate_restaurant_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_two_calls_produce_different_keys(self) -> None:
        assert generate_restaurant_key() != generate_restaurant_key()


class TestHashRestaurantKey:
    def test_same_key_produces_same_hash(self) -> None:
        key = "some-random-key"
        assert hash_restaurant_key(key) == hash_restaurant_key(key)

    def test_different_keys_produce_different_hashes(self) -> None:
        assert hash_restaurant_key("key-a") != hash_restaurant_key("key-b")

    def test_hash_is_not_the_plaintext_key(self) -> None:
        key = "some-random-key"
        assert hash_restaurant_key(key) != key

    def test_hash_is_a_64_char_hex_digest(self) -> None:
        digest = hash_restaurant_key("some-random-key")
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestVerifyRestaurantKey:
    def test_correct_key_against_its_own_hash_passes(self) -> None:
        key = generate_restaurant_key()
        stored_hash = hash_restaurant_key(key)
        assert verify_restaurant_key(key, stored_hash) is True

    def test_wrong_key_against_a_different_hash_fails(self) -> None:
        stored_hash = hash_restaurant_key(generate_restaurant_key())
        assert verify_restaurant_key("totally-wrong-key", stored_hash) is False

    def test_empty_key_fails(self) -> None:
        stored_hash = hash_restaurant_key(generate_restaurant_key())
        assert verify_restaurant_key("", stored_hash) is False
