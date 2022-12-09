import linked_list
import merge

ll = linked_list.LinkedList()
s = merge.Solution()

def add_count():
    name = input("Enter name: ")
    phone = input("Enter phone: ")
    address = input("Enter address: ")
    list1 = [name, phone, address]
    status = ll.addTail(list1)
    if status:
        print('[INFO] Insert success')
    else:
        print('[WARNING] The contact already exits')

def delete_count():
    name = input("Enter name: ")
    status = ll.deleteNode(name)
    if status:
        print('[INFO] Delete success')
    else:
        print('[WARNING] Name not found')

def search_count():
    name = input("Enter name: ")
    contact = ll.search(name)
    if contact is not None:
        print(contact[0], contact[1], contact[2])
    else:
        print("[WARNING] Contact not found")

def show_count():
    """show sorted contacts"""
    ll.listsort()
    ll.printList()

def sortage_data():
    ll.sortage()
    
def loadData():
    ll.csv_to_LinkedList()
    