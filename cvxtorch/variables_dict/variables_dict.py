class VariablesDict():
    """ Helper class that contains a dictionary from a non-constant leaf to an index in *args,
    with a safe add method.
    If a list is given, will create the dictionary in the order of the list."""
    def __init__(self, provided_vars_list: list=[]):
        self.vars_dict = dict()
        for var in provided_vars_list:
            self.add_var(var)
            
    def add_var(self, var):
        """
        var is expected to be either a cp.Variable or a cp.Parameter, but is not enforced.
        """
        if var not in self.vars_dict:
            self.vars_dict[var] = len(self.vars_dict)

    def has_type_in_keys(self, look_type: type):
        """
        This function returns True if one of the keys of this variable is of a certain type,
        and False otherwise
        """
        for var in self.vars_dict.keys():
            if isinstance(var, look_type):
                return True
        return False