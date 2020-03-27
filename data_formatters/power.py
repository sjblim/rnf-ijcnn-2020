"""
power.py


Created by limsi on 23/03/2020
"""


import pandas as pd
import sklearn.preprocessing

from data_formatters.base import GenericDataFormatter, InputTypes, DataTypes
import data_formatters.helpers as utils

class PowerFormatters(GenericDataFormatter):

    @property
    def column_definition(self):
        return self._column_definition

    def __init__(self, model_name):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = self.get_fixed_params()['total_time_steps']

        self._column_definition = [
            ('Target_active_power', DataTypes.REAL_VALUED, InputTypes.TARGET),
            ('Voltage', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('Global_intensity', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('Sub_metering_1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('Sub_metering_2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('Sub_metering_3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ("Global_reactive_power", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
            ('id', DataTypes.CATEGORICAL, InputTypes.ID),
            ('t', DataTypes.REAL_VALUED, InputTypes.TIME)
        ]
        if  "rnf" not in model_name:
            print("Adding in input col for {}".format(model_name))
            self._column_definition.append(('Global_active_power', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT))


    def split_data(self, df, valid_boundary=0.6, test_boundary=0.8):
        """Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split.
        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data
        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        T = len(df)
        valid_index = int(T*valid_boundary)
        test_index = int(T*test_boundary)
        train = df.iloc[:valid_index]
        valid = df.iloc[valid_index:test_index]
        test = df.iloc[test_index:]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Initialise scaler caches
        self._real_scalers = {}
        self._target_scaler = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):

            if len(sliced) >= self._time_steps:
                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                self._real_scalers[identifier] \
                    = sklearn.preprocessing.StandardScaler().fit(data)

                self._target_scaler[identifier] \
                    = sklearn.preprocessing.StandardScaler().fit(targets)

                print('target vol=', self._target_scaler[identifier].scale_)
            identifiers.append(identifier)

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

        # Extract identifiers in case required
        self.identifiers = identifiers

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Extract relevant columns
        column_definitions = self.get_column_definition()
        id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                    column_definitions)
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_col):

            # Filter out any trajectories that are too short
            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
                    sliced_copy[real_inputs].values)
                df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """

        if self._target_scaler is None:
            raise ValueError('Scalers have not been set!')

        column_names = predictions.columns

        df_list = []
        for identifier, sliced in predictions.groupby('identifier'):
            sliced_copy = sliced.copy()
            target_scaler = self._target_scaler[identifier]

            for col in column_names:
                if col not in {'forecast_time', 'identifier'}:
                    sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
            df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        return output

        # Default params

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 50,
            'num_epochs': 100,
            'multiprocessing_workers': 5
        }

        return fixed_params

