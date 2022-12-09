import merge
import prettytable as pt
import pandas as pd
import csv
class LinkedList(object):
    # Node class representing elements of linked list.
    class Node:
        def __init__(self, v, n=None):
            self.value = v
            self.next = n
            
    # Constructor of linked list.
    def __init__(self):
        self.head = None
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def addHead(self, value):
        self.head = self.Node(value, self.head)
        self.size += 1

    def addTail(self, value):
        if self.search(value[0]) is not None:
            return False
        if self.isEmpty():
            self.addHead(value)
            return True
        temp = self.head
        while temp.next != None:
            temp = temp.next
        temp.next = self.Node(value)
        self.size += 1
        return True

    def length(self):
        return self.size
    
    def printList(self):
        temp = self.head
        a = 1
        table = pt.PrettyTable(["No", "Name", "Phone", "Address"])
        while temp != None:
            """print pretty table"""
            table.add_row([a]+temp.value)
            temp = temp.next
            a += 1
        print(table)
    
    def deleteNode(self, delValue):
        #delete Information
        temp = self.head
        if self.isEmpty():
            return False
        if delValue == self.head.value[0]:
            self.head = self.head.next
            self.size -= 1
            return True
        while temp.next != None:
            if temp.next.value[0] == delValue:
                temp.next = temp.next.next
                self.size -= 1
                return True
            temp = temp.next
        return False

    def search(self, searchValue):
        temp = self.head
        while temp != None:
            if temp.value[0] == searchValue:
                return temp.value
            temp = temp.next
        return None

    def listsort(self):
        s = merge.Solution()
        self.head = s.sortList(self.head)
        
    def sortage(self):
        header_list = ["Name", "Phone", "Address"]
        data_list = []
        temp = self.head
        while temp != None:
            data_list.append(temp.value)
            temp = temp.next
        with open("PhoneBook_data.csv", mode = "w", encoding="utf-8-sig", newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(header_list)
            writer.writerows(data_list)
        
            
        
        
    def csv_to_LinkedList(self):
        csv_reader = csv.reader(open("PhoneBook_data.csv"))
        a = 1
        for line in csv_reader:
            if a == 1:
                a = a+1
                continue #pass the first line
            self.addTail(line)