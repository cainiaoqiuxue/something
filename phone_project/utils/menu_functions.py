from utils.link_list import LinkList

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
