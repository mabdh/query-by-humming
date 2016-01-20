
import pickle
import glob, os
import midi
from random import randint
import difflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from DBController import DBController

class NoteEvent:
    tickDuration =  0
    channel = 0
    data = 0
    def __init__(self, tickDuration, channel, data):
        self.tickDuration = tickDuration
        self.channel = channel
        self.data = data
    def getInfo(self):
        return ("tickDuration: %d, channel = %d, data = %d"%(self.tickDuration,self.channel,self.data))
        
def getNoteEvents(pattern):
    counter = 0
    listOfNoteEvents = []
    for track in pattern:
        for m in track:
            typeM = str(type(m))
            if(typeM.find('NoteOnEvent')!=-1):
                if(m.data[1]>0):
                    currentNote = NoteEvent(m.tick,m.channel,m.data[0])
                    listOfNoteEvents.append(currentNote)
                else:
                    #Assumption: MIDI file is a monophonic and one channel recording
                    currentNote = listOfNoteEvents[counter]
                    currentNote.tickDuration = m.tick - currentNote.tickDuration
                    counter+=1
    return listOfNoteEvents

def getNoteData(listOfNoteEvents):
    listOfNotesData = []
    for e in listOfNoteEvents:
        listOfNotesData.append(e.data)
    return listOfNotesData

def getUDS(listOfNoteData):
    pointer = 1
    listOfUDS = []
    uds='s'
    while(pointer<len(listOfNoteData)-1):
        if(listOfNoteData[pointer]>listOfNoteData[pointer-1]):
            uds='u'
        elif(listOfNoteData[pointer]<listOfNoteData[pointer-1]):
            uds='d'
        pointer+=1
        listOfUDS.append(uds)
    return listOfUDS

def pattern2UDS(pattern):
    listOfNoteEvents = getNoteEvents(pattern)
    listOfNoteData = getNoteData(listOfNoteEvents)
    listOfUDS = getUDS(listOfNoteData)
    udsString = ''.join(listOfUDS)
#    print (','.join(str(e) for e in listOfNoteData))
#    print (','.join(listOfUDS))
    return udsString

def convert_wav_to_midi(filename):
    # filename without extension
    filewav = filename + ".wav"
    filemidi = "midiFiles/" + filename + ".mid"
    os.system("./waon -i " + filewav + " -o " + filemidi + " -w 3 -n 4096 -s 2048")

def write_over_midifiles_to_db(dbController, midiFiles):
    id_file = 1
    dbController.start_over_table()
    for file in midiFiles:
        dataFullpath = dataDir + file
        pattern = midi.read_midifile(dataFullpath)
        udsString= pattern2UDS(pattern)
        # listOfUDSString.append(udsString)
        dbController.insert_new_uds_file(id_file, file, udsString)
        id_file = id_file + 1

def show_filenames_from_databases(dbController):
    fileLists = dbController.get_uds_file_list()
    print(" id\t | filename \t\t")
    for file in fileLists:
        print(" " + str(file[0]) + "\t | " + str(file[1]) )

def show_filenames_and_uds_from_databases(dbController):
    fileLists = dbController.get_uds_file_list()

    print(" id\t | filename \t\t")
    for file in fileLists:
        udsString = dbController.get_uds_string_from_id(file[0])
        print(" " + str(file[0]) + "\t | " + str(file[1]) + " \t| " + udsString[0])


if __name__ == "__main__":
    dataDir = '/Users/muhammadabduh/Documents/Research/query-by-humming/'
    os.chdir(dataDir)
    midiFiles = glob.glob("midiFile/*.mid")
    # udsFile = dataDir + 'listOfUDS.pickle'
    dbController = DBController('QBH')
    # listOfUDSString = []
    show_filenames_and_uds_from_databases(dbController)
    convert_wav_to_midi('classic')
# minLen = 5
# pointer = randint(0,len(listOfUDSString)-1)
# hum = listOfUDSString[pointer]
# #lenStr = randint(minLen,len(hum))
# #startIdx = randint(0,len(hum)-lenStr)

# # pointer = 45
# lenStr = 134
# startIdx = 86
# hum = hum[startIdx:startIdx+lenStr]
# print("Pointer: ",pointer)
# print("LenStr: ",lenStr)
# print("StartIdx: ",startIdx)
# print("OriginalString: ",listOfUDSString[pointer])
# print("Hum: ",hum)
# bestPartialRatio = -1
# bestRatio = -1
# idxMatch = -1
# counter = 0
# for uds in listOfUDSString:
#     print(counter)
#     partialRatio = fuzz.partial_ratio(hum,uds)
#     ratio = fuzz.ratio(hum,uds)
#     if(bestPartialRatio < partialRatio):
#         bestPartialRatio = partialRatio
#         bestRatio = ratio
#         idxMatch = counter
#     elif((bestPartialRatio == partialRatio) and (bestRatio < ratio)):
#         bestRatio = ratio
#         idxMatch = counter
#     counter+=1    
#     print("FullRatio:", ratio)
#     print("PartialRatio: ", partialRatio)
#     print("==============")
# if(idxMatch>=0):
#     print("IdxMatch: ", idxMatch)
#     print("FullUDS: ", listOfUDSString[idxMatch])
# process.extract(hum, listOfUDSString, limit=2)
# # best = difflib.get_close_matches(listOfUDSString[pointer], listOfUDSString)


