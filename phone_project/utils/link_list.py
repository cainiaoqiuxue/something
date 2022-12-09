class LinkList(object):
    def __init__(self, name=None, phone_number=None, address=None):
        self.name = name
        self.phone_number = phone_number
        self.address = address
        self.next = None

    def show(self):
        if self.name:
            print('-' * 50)
            print("Contact Person: ", self.name)
            print("Phone number: ", self.phone_number)
            print("Address: ", self.address)

    def save(self):
        res = {}
        head = self
        while head is not None:
            if head.name:
                res[head.name] = {"phone number": head.phone_number, "address": head.address}
            head = head.next
        res = str(res)
        with open('data/phone_data.txt', encoding='utf-8', mode='w') as f:
            f.write(res)