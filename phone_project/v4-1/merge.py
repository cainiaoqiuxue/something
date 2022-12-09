import linked_list
class Solution():
    """
    @param: head: The head of linked list.
    @return: You should return the head of the sorted linked list, using constant space complexity.
    """
    # mergesort
    def sortList(self, head):
        # write your code here
        if head is None or head.next is None:
            return head
        pre = head
        slow = head               # 使用快慢指针来确定中点
        fast = head
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        left = head  
        right = pre.next  
        pre.next = None           # 从中间打断链表
        left = self.sortList(left)  
        right = self.sortList(right)  
        return self.merge(left,right)
        
    def merge(self, left, right):
        pre = linked_list.LinkedList.Node(-1)
        first = pre
        while left and right:
            if ord(left.value[0].upper()[0]) < ord(right.value[0].upper()[0]):
                pre.next = left
                left = left.next
                pre = pre.next
            else:
                pre.next = right
                right = right.next
                pre = pre.next
        if left:
            pre.next = left
        else:
            pre.next = right
                
        return first.next

    def printList(self,head):
        temp = self.sortList(head)
        while temp != None:
            for i in temp.value:
                print("Name: ",i[1]),print("Phone: ",i[2]),print("Address: ",i[3])
            temp = temp.next
        print() 

