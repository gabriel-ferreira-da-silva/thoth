class RestrictedDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"'{key}' is not a valid setting.")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        raise KeyError(f"Deletion of keys is not allowed.")
