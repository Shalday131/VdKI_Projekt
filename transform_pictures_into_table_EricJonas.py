import image_to_table_EricJonas as itt
import cv2 as cv
import preimg_EricJonas as preimg

imginfo = itt.imgorga("Bilder")
imgextract = preimg.preimg()

feature_names = ["Rel_BreitGross", "RelSpitze_oben", "RelSpitze_unten", "Anzahl_Linie", "Anzahl_Ecken",
                 "Anzahl_Kreis", "Label"]

for idx, label in enumerate(imginfo.labels):
    for img_path in imginfo.image_paths[idx]:
        img = cv.imread(img_path)

        imgextract.setimg(img)
        imgextract.calc_canny_cnts()

        rel = imgextract.calc_rel_breitgroß_2()
        spitzo, spitzu = imgextract.calc_rel_spitze()
        line_count = imgextract.calc_canny_lines()
        edge_count = imgextract.calc_edges()
        circle_count = imgextract.calc_circles()

        imginfo.collect_numeric_data([rel, spitzo, spitzu, line_count, edge_count, circle_count, idx])

pd_imgdata = imginfo.get_feature_names(feature_names)

pd_imgdata.to_excel("Features.xlsx")
pd_imgdata.to_csv('Features.csv')

