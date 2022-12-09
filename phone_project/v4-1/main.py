import  function_meau as fm

def main():
  ans=True
  fm.loadData()
  while ans:
      print ("""
      ******* Phone Directory Management System *******
      1.Insert new records
      2.Delete existing records
      3.Search a record by name
      4.Display records in sorted order
      5.Quit the system
      """)
      ans=input("What would you like to do? ") 

      if ans =='1': 
        print("insert new records:")
        # Your code about how to insert new records #
        fm.add_count()
      
      elif ans == '2':
        print("delete records:")
        # Your code about how to delete existing records #
        fm.delete_count()
        
      elif ans == '3':
        print("search a record:")      
       # Your code about how to search a record by name #
        fm.search_count()

      elif ans == '4':
        print("display records in sorted order:")   
        # Your code about how to display records in sorted order#
        fm.show_count()
      elif ans == '5': 
        fm.sortage_data()
        break

      else: 
        print("Unknown Option Selected!")


if __name__ == '__main__':
    main()