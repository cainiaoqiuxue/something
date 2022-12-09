import os

from utils.link_list import LinkList
from utils.menu_functions import insert, show_all, delete

def main():
    with open('data/init_screen.txt', 'r') as f:
        init_screen = f.read()
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