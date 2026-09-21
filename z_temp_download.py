from roboflow import Roboflow

rf = Roboflow(api_key="puoOTsLc723eiaPHDWpD")
project = rf.workspace("ege-tanriverdi").project("scoliosis-detection-7bemw-sijcs")
version = project.version(1)
dataset = version.download("coco")
                