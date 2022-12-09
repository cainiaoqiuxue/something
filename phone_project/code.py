# -*- coding: utf-8 -*-
import os


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
        with open('./phone_data.txt', encoding='utf-8', mode='w') as f:
            f.write(res)


def check_duplicate(head, name):
    while head is not None:
        if name == head.name:
            return True
        else:
            head = head.next
    return False

def insert(head):
    name = input("Please type in name: ")
    if check_duplicate(head, name):
        print("The contact already exits")
        return head
    phone_number = input("Please type in phone number: ")
    address = input("Please type in address: ")
    point = head
    while point.next is not None:
        point = point.next
    point.next = LinkList(name, phone_number, address)
    print('insert success')
    return head

def show_all(head):
    while head is not None:
        head.show()
        head = head.next

def delete(head):
    name = input('Please type in name: ')
    point = head
    while point.next is not None:
        if point.next.name == name:
            point.next = point.next.next
            print('delete success')
            return head
        else:
            point = point.next
    print('name not found')


def main():
    init_screen = '''
    The system provide the functions as follow
    1: Insert
    2: Delete
    3: Revise
    4: Search
    5: Quit
    6: Display all the records

'''
    head = LinkList()
    while True:
        print('\n')
        print(init_screen)
        choose = input('Please enter the corresponding number to operate: ')
        if choose == '1':
            head = insert(head)
        elif choose == '2':
            head = delete(head)
        elif choose == '5':
            head.save()
            os.system("cls")
            break
        elif choose == '6':
            show_all(head)
        else:
            print('invalid input')

if __name__ == '__main__':
    main()