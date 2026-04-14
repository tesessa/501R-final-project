class Struct():
    def __init__(self, **entries):
        self.config_dict = entries
        for key, value in entries.items():
            setattr(self, key, value)

    def __str__(self):
        s = "Struct: {"
        for key, value in self.config_dict.items():
            s += f"{key}: {value},"
        s = s[:-1] + "}"
        return s

    def get_config_dict(self):
        return self.config_dict