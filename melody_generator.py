import json
import numpy as np
import music21 as m21
import tensorflow
keras=tensorflow.keras
SAVE_MODEL_PATH_BEFORE="model_before.h5"
SAVE_MODEL_PATH_AFTER="model_after.h5"

from preprocess import SEQUENCE_LENGTH,MAPPING_PATH

class melody_generator:

    def __init__(self, model_path="model.h5"):
        
        self.model_path=model_path
        self.model=keras.models.load_model(model_path)
        self.model_after=keras.models.load_model(SAVE_MODEL_PATH_AFTER)
        self.model_before=keras.models.load_model(SAVE_MODEL_PATH_BEFORE)

        with open(MAPPING_PATH,"r") as fp:
            self._mappings=json.load(fp)

        self._start_symbols=["/"]*SEQUENCE_LENGTH

#---------------------

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        #seed is the string we r working on

        #create a seed with start symbols
        seed =seed.split()
        melody=seed
        seed=self._start_symbols+seed

        #map seed to int
        seed=[self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            #just see the last max_sequence_length inputs
            seed=seed[-max_sequence_length:]
            seed_A=[]
            seed_B=[]

            for i,seedling in enumerate(seed):
                if seedling==0:
                    seed_B.append(seedling)
                    seed_A.append(seedling+1)
                elif seedling==37:
                    seed_B.append(seedling-1)
                    seed_A.append(seedling)
                else:
                    seed_A.append(seedling+1)
                    seed_B.append(seedling-1)

            #one-hot encode the seed
            onehot_seed=keras.utils.to_categorical(seed,num_classes=len(self._mappings))
            onehot_seed_A=keras.utils.to_categorical(seed_A,num_classes=len(self._mappings))
            onehot_seed_B=keras.utils.to_categorical(seed_B,num_classes=len(self._mappings))
            #now we have a matrix with dimensions
            #max-sequence_length x len(self._mappings)
            #but this is not acceptable by predict functon of tensor flow, it only works with 3D arrays, so we gotta add another dimension

            onehot_seed=onehot_seed[np.newaxis,...]
            onehot_seed_A=onehot_seed_A[np.newaxis,...]
            onehot_seed_B=onehot_seed_B[np.newaxis,...]
            #1 x max-sequence_length x len(self._mappings)

            probabilities=self.model.predict(onehot_seed)[0]
            probabilities_A=self.model_after.predict(onehot_seed_A)[0]
            probabilities_B=self.model_before.predict(onehot_seed_B)[0]
            #as we use a single dimension(newly created one)
            #probabilities will look like
            #[0.1,0.2,0.1,0.6]->1

            output_int=self._sample_with_temperature(probabilities, temperature)
            output_int_A=self._sample_with_temperature(probabilities_A, temperature)
            output_int_B=self._sample_with_temperature(probabilities_B, temperature)

            #update seed
            if output_int_B==0 or output_int_A==37:
                seed.append(output_int)
            elif output_int_A-1==output_int_B+1:
                seed.append(output_int_A-1)
            else:
                seed.append(output_int)

            #update seed
            #seed.append(output_int)

            #map int to our coding, going in reverse
            output_symbol=[key for key, value in self._mappings.items() if value==output_int][0]

            #check whether we're at the end of the melody
            if output_symbol=="/":
                break

            #update the melody
            melody.append(output_symbol)

        return melody


#---------------------

    def _sample_with_temperature(self, probabilities, temperature):
        #lets understand temprature and why we wanna use it
        #intutivly, temp->infinity, our probabilities curve will get flatter
        # likewise, if temp ->0, our probability curve will show more depiraty
        #this helps us be more precise as when we are picking a random choice based on probablities, we are more/less(choosable) likely to get the one we want

        predictions=np.log(probabilities)/temperature
        probabilities=np.exp(predictions)/np.sum(np.exp(predictions))
        #now we have probabilities between 0-1

        choices=range(len(probabilities))
        index=np.random.choice(choices,p=probabilities)

        return index

#--------------------

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.midi"):
        #create a m21 stream
        stream=m21.stream.Stream()

        #parse all the symbols in the melody and create note/rests objects
        start_symbol=None
        step_counter=1

        for i, symbol in enumerate(melody):
            #handle notes/rests
            if symbol!="_" or i+1==len(melody):

                #to ensure we are not dealing with the first note/rest
                if start_symbol is not None:
                    quarter_length_duration=step_duration*step_counter                    

                    #handle rests
                    if start_symbol =="r":
                        m21_event=m21.note.Rest(quarterLength=quarter_length_duration)

                    #handle notes
                    else:
                        m21_event=m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)
                    
                    #reset the step counter
                    step_counter=1

                start_symbol=symbol

            #handle prologation "_"
            else:
                step_counter+=1

        #write the m21 stream to midi file
        stream.write(format,file_name)

#--------------------

if __name__=="__main__":
    mg=melody_generator()
    seed1="67 _ _ _ 67 _ _ _ 64 _ _ _ 60 _ _ _ 67 _ _ _ _ "
    seed2="67 _ _ _ 67 _ _ _ _ _ _ _ 70 _ _ _ _ _ _ _ 69 _ _ _ _ _ _ _ 72 _ _ _ _ _ _ _ 70 _ _ _ _ _"

    melody=mg.generate_melody(seed1, 500, SEQUENCE_LENGTH, 1)

    print(melody)

    mg.save_melody(melody)