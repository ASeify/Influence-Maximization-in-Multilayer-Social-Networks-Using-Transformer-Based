# version: 1.0.3
from tkinter import Tk, filedialog
from contextlib import contextmanager
import os
from pathlib import Path


class Files_Handler:
    @staticmethod
    def select_files(file_type:'str', visible_types:'str'='', win_title:str='Select File', multiple_mod:'bool'=False):
        try:
            root = Tk()
            root.withdraw()
            root.call("wm", "attributes", ".", "-topmost", True)
            if visible_types == '':
                visible_types = '.*'
            file_path = filedialog.askopenfilename(
                filetypes=[(file_type, visible_types)],
                title=win_title,
                multiple=multiple_mod)
            return file_path
        except Exception as e:
            print(e)
    
    @staticmethod
    def select_dir(win_title:str='Select Directory'):
        path = None
        try:
            root = Tk()
            root.withdraw()
            root.call("wm", "attributes", ".", "-topmost", True)
            path = filedialog.askdirectory(title=win_title)
            return path
        except Exception as e:
            return path

    @staticmethod
    def get_files_in_path(path:'str', types:str | list[str] = '*'):
        files = []
        files_list = next(os.walk(path), (None, None, []))[2]
        if types == '*':
            return files_list
        else:
            for item in files_list:
                if item.split(".")[-1] in types:
                    files.append(item)
        return files
    
    @staticmethod
    def make_dir(root_path: str, dir_name: str):
        new_path = root_path + dir_name
        if os.path.exists(new_path):
            return new_path + "/"
        try:
            os.mkdir(new_path)
            return new_path + "/"
        except:
            return False
    
    @staticmethod
    def create_new_file(path:str, name:str, ext:str, mode:str):
        if path[-1] != "/":
            path += "/"
            mode = mode
        return open(str(path + name + ext), mode)
    
    @staticmethod
    def get_file_info(inp_file):
        file_path = str(inp_file)
        info = {}
        info['path'] = file_path[:file_path.rfind("/")] + "/"
        name_type = file_path.split(".")
        type = "." + name_type[-1]
        if len(name_type) > 2:
            name_type = file_path.split("/")[-1]
            info['name'] = name_type[:name_type.rfind(".")]
        else:
            info['name'] = name_type[0].split("/")[-1]
        info['type'] = type
        # info['mode'] = inp_file.mode
        
        return info
   
    @staticmethod
    def find_file_in_dir(root, file_name, file_type:str='*'):
        files = Files_Handler.get_files_in_path(root, file_type)
        if len(files) > 0:
            for item in files:
                item_info = Files_Handler.get_file_info(item)
                if file_type == "*":
                    if item_info['name'] == file_name:
                        return item_info
                else:
                    if item_info['name'] == file_name and item_info['type'] == file_type:
                        return item_info

        return False

    @staticmethod
    def get_file_path_info(file_path:str):
        info = {}
        info['path'] = file_path[:file_path.rfind("/")] + "/"
        name_type = file_path.split(".")
        type = "." + name_type[-1]
        if len(name_type) > 2:
            name_type = file_path.split("/")[-1]
            info['name'] = name_type[:name_type.rfind(".")]
        else:
            info['name'] = name_type[0].split("/")[-1]
        info['type'] = type        
        return info
    
    @staticmethod
    def get_dirs_in_path(file_path:str):
        dir_list = [x[0] for x in os.walk(file_path)]
        return dir_list
    
    @staticmethod
    @contextmanager
    def tk_windows_timer(timeout=600):
        root_timer = Tk() # default root
        root_timer.withdraw() # remove from the screen

        # destroy all widgets in `timeout` seconds
        func_id = root_timer.after(int(1000*timeout), root_timer.quit)
        try:
            yield root_timer
        finally: # cleanup
            root_timer.after_cancel(func_id) # cancel callback
            root_timer.destroy()
    
    @staticmethod
    def get_files_by_extensions(folder_path: str, extensions: list[str]) -> list[str]:
        """
        Recursively retrieves all files in `folder_path` and subfolders that match the given extensions.
        
        Args:
            folder_path (str): The path to the root folder.
            extensions (List[str]): List of file extensions to match (e.g., ['.txt', '.jpg']).
        
        Returns:
            List[str]: List of matching file paths as strings.
        """
        path_obj = Path(folder_path)
        matched_files = []

        for ext in extensions:
            files_path = path_obj.rglob(f'*{ext}')
            for file_path in files_path:
                matched_files.append(str.replace(str(file_path),'\\', '/'))  # Normalize path separators

        return [str(f) for f in matched_files]
    pass