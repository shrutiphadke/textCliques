# Copyright(C) Shruti Phadke
from laserembeddings import Laser


def calculate_multiling_laser(textCorp, langCodes):
    laser = Laser()
    embeddings = laser.embed_sentences(textCorp, lang=langCodes)
    return embeddings