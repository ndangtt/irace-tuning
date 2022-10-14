import ConfigSpace
import numpy as np
from ConfigSpace.util import fix_types, impute_inactive_values
from ConfigSpace.hyperparameters import NumericalHyperparameter

def convert_params_to_vec(params, cs):
    """
    Convert dict - params in list representation to a numerical array
    representation.
    The types of the parameters are determined by the configuration space.
    The returned array is in the same form and order as the data the surrogate
    model has been trained on.

    Parameters
    ----------
    params : list(str)
        List of parameters in form '--name value' or '-name value'.
    cs : ConfigSpace.configuration_space
        Contains the information about the parameters from `params`. Parameter
        in `cs` and `params` must match!
        
    Returns
    -------
        np.array
            array representation of the parameters.
    """
    config = get_imputed_config_from_dict(config=params,
                                          cs=cs,
                                          impute_with='def')

    config = encode_config_as_array_with_true_values(config=config,
                                                        cs=cs)
    return config

def get_imputed_config_from_dict(config, cs, impute_with='default'):
    """
    Create a configuration from a dictionary. A configuration created with the
    old configuration space module (2015 from aclib),
    may contain inactive hyperparameters. Therefore, some preprocessing is
    necessary.

    Parameters
    ----------
config : dict
        dictionary representation of a ConfigSpace.Configuration
    cs : ConfigSpace.ConfigurationSpace

    impute_with : str, optional
        imputation strategy. Defaults to 'def'

    Returns
    -------
    ConfigSpace.Configuration
    """

    config_dict = get_imputed_config_as_dict_from_dict(config, cs, impute_with)

    # Allow temporarily inactive values to allow the building of the
    # configuration object. Otherwise a error will be raised.
    # But after creating the configuration, the inactive parameters will be
    # imputed (similar to old config space).
    config = ConfigSpace.Configuration(configuration_space=cs,
                              values=config_dict,
                              allow_inactive_with_values=True)

    # TODO: These steps are unnecessary! (exact same behaviour like above, but
    #       for dictionaries.
    # make sure it works with new configspace ('def'--> 'default')
    impute_with = 'default' if impute_with == 'def' else impute_with
    config = impute_inactive_values(config, impute_with)

    return config

def get_imputed_config_as_dict_from_dict(config, cs, impute_with='def'):
    """
    Create a configuration in dictionary representation from a dictionary.
    A configuration created with the old configuration space module
    (2015 from aclib), may contain inactive hyperparameters.
    Therefore, some preprocessing is necessary.

    Parameters
    ----------
    config : dict
        dictionary representation of a ConfigSpace.Configuration
    cs : ConfigSpace.ConfigurationSpace

    impute_with : str,int,float
        imputation strategy. Defaults to 'def'

    Returns
    -------
    ConfigSpace.Configuration
    """
    config_dict = fix_types(config, cs)
    # include missing (inactive parameters)
    if impute_with == 'def':
        config_dict = \
            {name:
             config_dict.get(name, cs.get_hyperparameter(name).default_value)
             for name in cs.get_hyperparameter_names()}
    if type(impute_with) in [int, float, np.float, np.int]:
        config_dict = \
            {name:
             config_dict.get(name, impute_with)
             for name in cs.get_hyperparameter_names()}

    return config_dict

def encode_config_as_array_with_true_values(config, cs, normalize=False):
    """
    Method to produce same array representation as the old configspace module
    from the aclib. It is similar to the array representation of the
    configspace module, but continuous hyperparameter are not scaled between
    0 and 1, but instead replaced by their "true" value.

    Note: In the ConfigSpace package, categorical hyperparameter are replaced by
          the indices of the value in its choices.

    Parameters
    ----------
    config : configuration
    cs : ConfigSpace object
    normalize : bool, optional
        If True, all values are scaled between 0 and 1. categorical values are
        simply the index of their values in the hyperparameters choices.

    Returns
    -------
    np.array
        Array Representation of the configuration as describe above
    """
    cfg_array = config.get_array()

    if normalize:
        return cfg_array

    # This is not necessary for inactive parameter, because for they already
    # default value is used
    active_hp_names = [name
                       for name in cs.get_active_hyperparameters(config)]
    inactive_hp_names = set(cs.get_hyperparameter_names()) \
                        - set(active_hp_names)

    for i, hp in enumerate(cs.get_hyperparameters()):
        if hp.name in inactive_hp_names:
            continue

        if issubclass(type(hp), NumericalHyperparameter):
            value = config[hp.name]
            value = np.log(value) if hp.log else value
            cfg_array[i] = value

    return cfg_array
