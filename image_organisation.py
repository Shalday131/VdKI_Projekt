import os
import cv2 as cv

class image_organisation:

    def __init__(self):

        self.labels = []
        self.image_paths = []
        self.images = []

        for label_name in os.listdir("Bilder"):                     # schreibt Namen der Ordner und Dateien, die sich unter "Bilder" befinden in label_name
            label_path = os.path.join("Bilder", label_name)         # hängt den label_name an den Pfad "Bilder" ran und generiert den Pfad für alles was sich unter "Bilder" befindet
            if os.path.isdir(label_path):                           # überprüft ob die einzelnen Pfade zu einem Ordner führen
                self.labels.append(label_name)                      # fügt den Namen des Ordners in das Array self.labels hinzu
            for image_name in os.listdir(label_path):               # schreibt Namen der Ordner und Dateien, die sich unter label_path befinden in image_name
                image_path = os.path.join(label_path, image_name)   # hängt den image_name an label_path an und generiert Pfad für Datein und Odner darunter
                if os.path.isfile(image_path):                      # überprüft ob image_path zu einer Datei führt
                    self.image_paths.append(image_path)             # hängt Pfad der Datei an image_paths an

    def get_images(self):
        self.images.clear()
        for image_path in self.image_paths:
            self.images.append(cv.imread(image_path))
        return self.images

    def print(self):
        print(self.labels)
        print(self.image_paths)
        print(self.images)