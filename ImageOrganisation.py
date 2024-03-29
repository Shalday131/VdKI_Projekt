# Beschreibung der Funktionen
# Diese Klasse wird verwendet um den Pfad von Bildern auszulesen. Zusätzlich wird hier der Ordnername, in dem
# sich die Bilder befinden gespeichert. Des Weiteren werden hier die ausgelesenen Features in eine Excel-Datei
# geschrieben.
#
# Autoren: Barabanow, Günter, Kauff, Sachweh

import os
import cv2 as cv
import pandas as pd

class ImageOrganisation:

    def __init__(self):
        self.labels = []
        self.image_paths = []
        self.images = []
        self.label_per_image = []
        self.data_list = []

        for label_name in os.listdir("E:\Bilder VdKI\Bilder 3"):                     # schreibt Namen der Ordner und Dateien, die sich unter "Bilder" befinden in label_name
            label_path = os.path.join("E:\Bilder VdKI\Bilder 3", label_name)         # hängt den label_name an den Pfad "Bilder" ran und generiert den Pfad für alles was sich unter "Bilder" befindet
            if os.path.isdir(label_path):                           # überprüft ob die einzelnen Pfade zu einem Ordner führen
                self.labels.append(label_name)                      # fügt den Namen des Ordners in das Array self.labels hinzu
            for image_name in os.listdir(label_path):               # schreibt Namen der Ordner und Dateien, die sich unter label_path befinden in image_name
                image_path = os.path.join(label_path, image_name)   # hängt den image_name an label_path an und generiert Pfad für Datein und Odner darunter
                if os.path.isfile(image_path):                      # überprüft ob image_path zu einer Datei führt
                    self.image_paths.append(image_path)             # hängt Pfad der Datei an image_paths an

    def get_images(self):                                           # Funktion gibt Bilder in einem Array zurück
        self.images.clear()                                         # falls die Funktion öfters aufgerufen wird, wird der Array zuerst gecleert
        for image_path in self.image_paths:                         # iteriere durch alle image_paths
            self.images.append(cv.imread(image_path))               # lese Bild von image_path ein und hänge das Bild an self.images an
        print("Anzahl der eingelesenen Bilder: ", len(self.images))
        return self.images                                          # gibt das Array self.images zurück

    def get_label(self, image_path):
        label = os.path.basename(os.path.dirname(image_path))
        return label

    # schreibt die gefundenen Features in eine Liste
    def collect_features(self, data):
        self.data_list.append(data)

    # schreibt die gesammelten Daten in ein Dataframe
    def get_dataframe(self, column_names):
        df = pd.DataFrame(self.data_list, columns=column_names)
        return df
