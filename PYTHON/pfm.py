'''
Author: Philipp Fahler-Muenzer
Python version: 3.11.7
'''

def save_Object(obj, filename, path=''):
        '''
        Saves an Object in the path
        '''
        #Check wether path is empty == if yes, try to split up the filename
        if path == '':
            filename, path = POLDERs_data.seperate_file_folder(filename)
        
        #If the path is not empty, check wether path exists and if not, create it!
        if not path=='':
            if not os.path.exists(path):
                os.makedirs(path)
        
        #Now the file is saved!
        print('Saving file: '+filename)
        with open(os.path.join(path, filename),'wb') as f: 
            pickle.dump(obj, f)
        print('Done!')