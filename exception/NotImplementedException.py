class NotImplementedException(Exception):
    def __init__(self, func, cls):
        self.func = func
        self.cls = cls
        super().__init__(f"Function {func} (of class {cls}) is not implemented!")
