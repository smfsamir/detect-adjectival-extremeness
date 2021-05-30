class Optional():
    def __init__(self, value=None): 
        if value:
            self.exists = True
            self.val = value 
        else:
            self.exists = False
            self.val = None

    def __repr__(self):
        return f"Optional({self.val})" if self.exists else "Optional()"