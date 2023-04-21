import pandas as pd
import numpy as np
import pytest
from house_price_prediction import create_dummies_from_list

class TestCreateDummiesFromList:

    def test_basic_functionality(self):
        df = pd.DataFrame({
            "animal": ["dog", "cat", "bird", "dog", "cat"],
        })
        dummies = pd.DataFrame(["animal_dog", "animal_cat", "animal_bird", "animal_other"])

        result = create_dummies_from_list(df, "animal", dummies)

        expected = pd.DataFrame({
            "animal_dog": [1, 0, 0, 1, 0],
            "animal_cat": [0, 1, 0, 0, 1],
            "animal_bird": [0, 0, 1, 0, 0],
            "animal_other": [0, 0, 0, 0, 0],
        }, dtype=np.uint8)

        pd.testing.assert_frame_equal(result, expected)

    def test_with_extra_columns(self):
        df = pd.DataFrame({
            "animal": ["dog", "cat", "bird", "dog", "cat", "elephant"],
        })
        dummies = pd.DataFrame(["animal_dog", "animal_cat", "animal_bird", "animal_other"])
        result = create_dummies_from_list(df, "animal", dummies)

        expected = pd.DataFrame({
            "animal_dog": [1, 0, 0, 1, 0, 0],
            "animal_cat": [0, 1, 0, 0, 1, 0],
            "animal_bird": [0, 0, 1, 0, 0, 0],
            "animal_other": [0, 0, 0, 0, 0, 1],
        }, dtype=np.uint8)

        pd.testing.assert_frame_equal(result, expected)

    def test_with_missing_columns(self):
        df = pd.DataFrame({
            "animal": ["dog", "cat", "dog", "cat"],
        })
        dummies = pd.DataFrame(["animal_dog", "animal_cat", "animal_bird", "animal_other"])
        result = create_dummies_from_list(df, "animal", dummies)

        expected = pd.DataFrame({
            "animal_dog": [1, 0, 1, 0],
            "animal_cat": [0, 1, 0, 1],
            "animal_bird": [0, 0, 0, 0],
            "animal_other": [0, 0, 0, 0],
        }, dtype=np.uint8)

        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        dummies = pd.DataFrame(["animal_dog", "animal_cat", "animal_bird", "animal_other"])
        with pytest.raises(ValueError):
            create_dummies_from_list(df, "animal", dummies)
