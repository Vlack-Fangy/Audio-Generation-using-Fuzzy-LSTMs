import os,pathlib,json
import music21 as m21
import numpy as np
import tensorflow as tensorflow
keras=tensorflow.keras
#instead of tensorflow.Keras as Keras

ACCEPTABLE_DURATIONS=[
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]
SEQUENCE_LENGTH=64

KERN_DATASET_PATH="deutschl/essen/europa/deutschl/erk"
SAVE_DIR='dataset'
SINGLE_FILE_DATASET="file_dataset.txt"
MAPPING_PATH='mapping.json'

def load_songs_in_kern(dataset_path):
    #go through all the files in dataset and load them with music21
    songs=[]
    #i=0

    for path,subdirs,files in os.walk(dataset_path):
        for file in files:
            if file[-3:]=="krn":
                song=m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs

#-----------------------

def has_acceptable_durations(song,acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

#----------------------

def transpose(song):
    #get key from the Song
    parts=song.getElementsByClass(m21.stream.Part)
    measure_part0=parts[0].getElementsByClass(m21.stream.Measure)
    key=measure_part0[0][4]   

    if not isinstance(key,m21.key.Key):
        key=song.analyze("Key")
    
    #print(song.analyze("Key"))-why does it not work---doubt 1 

    #get interval from transposition, e.g. Bmaj->Cmaj
    if key.mode=="major":
        interval=m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))
    elif key.mode=="minor":
        interval=m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))

    #print(interval)

    #transpose song by calculated interval
    transposed_song=song.transpose(interval)
    #print(transposed_song)

    return transposed_song

#-----------------------

def encode_song(song,time_step=0.25):
    # p=60,d=1.0->[60,"_","_","_"]

    encoded_song=[]

    for event in song.flat.notesAndRests:

        #handle notes
        if isinstance(event,m21.note.Note):
            symbol=str(event.pitch.midi)
        #handle rests
        elif isinstance(event,m21.note.Rest):
            symbol='r'

        #convert the note/rest into time series notation
        steps=int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')

    #cast song in a str
    encoded_song=" ".join(encoded_song)

    return encoded_song

#-----------------------

def preprocess(dataset_path):
    #load the folk songs
    print("Loading Songs...")
    songs=load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs")

    for i, song in enumerate(songs):
        #filter out the songs that have non-acceptable durations
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue

        #transpose songs to Cmaj/Amin
        song=transpose(song)

        #encode songs with music time series representation
        encoded_song=encode_song(song)

        #save songs to text file
        save_path=os.path.join(SAVE_DIR, str(i)+'.txt')
        with open(save_path,"w") as fp:
            fp.write(encoded_song)
        
    

#-----------------------------

def load(file_path):
    with open(file_path) as fp:
        song=fp.read()
    return song

#-----------------------------

def create_single_file_dataset(dataset_path,file_dataset_path,sequence_length):
    new_song_delimiter='/ '*sequence_length
    songs=""

    #load encoded songs and add delimiters
    for path,_,files in os.walk(dataset_path):
        for file in files:
            file_path=os.path.join(path,file)
            song=load(file_path)
            songs+=song+' '+new_song_delimiter

    songs=songs[:-1]

    #save string that contains all the dataset
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)
    
    return songs


#-----------------------------

def create_mapping(songs,mapping_path):
    mapping={}

    #identify the vocabulary
    songs=songs.split()
    vocabulary=list(set(songs))
    vocabulary.sort()

    vocabulary=vocabulary[:-2]
    vocabulary.insert(1,"r")
    vocabulary.insert(1,"_")

    #create mapping
    for i,symbol in enumerate(vocabulary):
        mapping[symbol]=i

    #save vocabulary to a json file
    with open(mapping_path,"w") as fp:
        json.dump(mapping,fp,indent=4)
    

#-----------------------------

def convert_songs_to_int(songs):
    int_songs=[]

    #load mappings
    with open(MAPPING_PATH,"r") as fp:
        mappings=json.load(fp)

    #cast songs string to a list
    songs=songs.split()

    #map Songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

#-----------------------------

def generating_training_sequences(sequence_length):

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    int_songs_B=[]
    int_songs_A=[]

    for i, num in enumerate(int_songs):
        if num==0:
            int_songs_A.append(num+1)
            int_songs_B.append(num)
        elif num==37:
            int_songs_A.append(num)
            int_songs_B.append(num-1)
        else:
            int_songs_A.append(num+1)
            int_songs_B.append(num-1)

    print(type(int_songs))
    print(len(int_songs))
    print(type(int_songs_A))
    print(len(int_songs_A))

    inputs = []
    targets = []

    inputs_B=[]
    targets_B=[]

    inputs_A=[]
    targets_A=[]

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

        inputs_B.append(int_songs_B[i:i+sequence_length])
        targets_B.append(int_songs_B[i+sequence_length])

        inputs_A.append(int_songs_A[i:i+sequence_length])
        targets_A.append(int_songs_A[i+sequence_length])


    #now, to break stuff into training and testing
    inputs_train=inputs[:int(num_sequences*0.8)]
    inputs_test=inputs[int(num_sequences*0.8):]

    targets_train=targets[:int(num_sequences*0.8)]
    targets_test=targets[int(num_sequences*0.8):]

    inputs_B_train=inputs_B[:int(num_sequences*0.8)]
    inputs_B_test=inputs_B[int(num_sequences*0.8):]

    targets_B_train=targets_B[:int(num_sequences*0.8)]
    targets_B_test=targets_B[int(num_sequences*0.8):]

    inputs_A_train=inputs_A[:int(num_sequences*0.8)]
    inputs_A_test=inputs_A[int(num_sequences*0.8):]

    targets_A_train=targets_A[:int(num_sequences*0.8)]
    targets_A_test=targets_A[int(num_sequences*0.8):]

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)

    inputs_train = keras.utils.to_categorical(inputs_train, num_classes=vocabulary_size)
    inputs_test = keras.utils.to_categorical(inputs_test, num_classes=vocabulary_size)
    
    inputs_B_train=keras.utils.to_categorical(inputs_B_train, num_classes=vocabulary_size)
    inputs_B_test=keras.utils.to_categorical(inputs_B_test, num_classes=vocabulary_size)

    #print(inputs_A_train)
    for i in inputs_A_train:
        for j in i:
            if j>37:
                print(i,j, sep=" ")
                break
    #how am I producing 38 0r more

    inputs_A_train=keras.utils.to_categorical(inputs_A_train, num_classes=vocabulary_size)
    inputs_A_test=keras.utils.to_categorical(inputs_A_test, num_classes=vocabulary_size)

    targets_train = np.array(targets_train)
    targets_test = np.array(targets_test)

    targets_A_train=np.array(targets_A_train)
    targets_A_test=np.array(targets_A_test)

    targets_B_train=np.array(targets_B_train)
    targets_B_test=np.array(targets_B_test)

    #print(f"There are {len(inputs)} sequences.")

    return inputs_B_train, targets_B_train, inputs_B_test, targets_B_test, inputs_train, targets_train, inputs_test, targets_test, inputs_A_train, targets_A_train, inputs_A_test, targets_A_test

#-----------------------------


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    
    inputs_B_train, targets_B_train, inputs_B_test, targets_B_test, inputs_train, targets_train, inputs_test, targets_test, inputs_A_train, targets_A_train, inputs_A_test, targets_A_test = generating_training_sequences(SEQUENCE_LENGTH)

if __name__=='__main__':
    main()
