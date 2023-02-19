import tensorflow as tensorflow
keras=tensorflow.keras
from preprocess import generating_training_sequences, SEQUENCE_LENGTH
from sklearn.metrics import mean_squared_error as r2
from melody_generator import melody_generator
import numpy as np

OUTPUT_UNITS=38 
#change it to the length of json element in mapping.json
NUM_UNITS=[256]
LOSS="sparse_categorical_crossentropy"
LEARNING_RATE=0.001
EPOCHS=10
BATCH_SIZE=64
SAVE_MODEL_PATH_MAIN="model.h5"
SAVE_MODEL_PATH_BEFORE="model_before.h5"
SAVE_MODEL_PATH_AFTER="model_after.h5"

INPUTS_TRAIN=[]
TARGETS_TRAIN=[]

INPUTS_TEST=[]
TARGETS_TEST=[]

INPUTS_B_TRAIN=[]
TARGETS_B_TRAIN=[]

INPUTS_B_TEST=[]
TARGETS_B_TEST=[]

INPUTS_A_TRAIN=[]
TARGETS_A_TRAIN=[]

INPUTS_A_TEST=[]
TARGETS_A_TEST=[]


#---------------------------

def _sample_with_temperature(prob,temp):
    pred=np.log(prob)/temp
    prob=np.exp(pred)/np.sum(np.exp(pred))
    c=range(len(prob))
    index=np.random.choice(c,p=prob)
    return index


#---------------------------

def build_model(output_units, num_units, loss, learning_rate):
    #create model architecture
    input=keras.layers.Input(shape=(None,output_units))         #basic structure with just Vocabulary length of neurons
    x=keras.layers.LSTM(num_units[0])(input)                    #adding a hidden layer(an LSTM layer) to the system, of length=256?
    x=keras.layers.Dropout(0.2)(x)                              #not for adding another layer, but just to combact overfitting

    output=keras.layers.Dense(output_units,activation="softmax")(x)         #addes the output layer

    model=keras.Model(input,output)

    #compile model
    model.compile(loss=loss,
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])

    model.summary()

    return model

#---------------------------

def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
    #generate the training sequences
    inputs_train=INPUTS_TRAIN
    targets_train=TARGETS_TRAIN
    
    #build the network
    model=build_model(output_units, num_units, loss, learning_rate)
    #model_before=build_model(output_units, num_units, loss, learning_rate)
    #model_after=build_model(output_units, num_units, loss, learning_rate)

    #train the model
    model.fit(inputs_train, targets_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #model_before.fit(inputs_before_train, targets_before_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #model_after.fit(inputs_after_train, targets_after_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    #save the model
    model.save(SAVE_MODEL_PATH_MAIN)
    #model.save(SAVE_MODEL_PATH_BEFORE)
    #model.save(SAVE_MODEL_PATH_AFTER)

def train_A(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
    #generate the training sequences
    inputs_train=INPUTS_A_TRAIN
    targets_train=TARGETS_A_TRAIN
    
    #build the network
    model=build_model(output_units, num_units, loss, learning_rate)
    #model_before=build_model(output_units, num_units, loss, learning_rate)
    #model_after=build_model(output_units, num_units, loss, learning_rate)

    #train the model
    model.fit(inputs_train, targets_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #model_before.fit(inputs_before_train, targets_before_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #model_after.fit(inputs_after_train, targets_after_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    #save the model
    #model.save(SAVE_MODEL_PATH_MAIN)
    #model.save(SAVE_MODEL_PATH_BEFORE)
    model.save(SAVE_MODEL_PATH_AFTER)

def train_B(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
    #generate the training sequences
    inputs_train=INPUTS_B_TRAIN
    targets_train=TARGETS_B_TRAIN
    
    #build the network
    model=build_model(output_units, num_units, loss, learning_rate)
    #model_before=build_model(output_units, num_units, loss, learning_rate)
    #model_after=build_model(output_units, num_units, loss, learning_rate)

    #train the model
    model.fit(inputs_train, targets_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #model_before.fit(inputs_before_train, targets_before_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #model_after.fit(inputs_after_train, targets_after_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    #save the model
    #model.save(SAVE_MODEL_PATH_MAIN)
    #model.save(SAVE_MODEL_PATH_BEFORE)
    model.save(SAVE_MODEL_PATH_AFTER)

#-------------------------

# def test(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
#     #generate the training sequences
#     inputs_before_train, inputs_before_test , targets_before_train, targets_before_test, inputs_train, inputs_test, targets_train, targets_test , inputs_after_train, inputs_after_test, targets_after_train, targets_after_test=generating_training_sequences(38)

#     model_path=SAVE_MODEL_PATH_MAIN
#     model=keras.models.load_model(SAVE_MODEL_PATH_MAIN)

#     model_path=SAVE_MODEL_PATH_BEFORE
#     model_before=keras.models.load_model(SAVE_MODEL_PATH_BEFORE)

#     model_path=SAVE_MODEL_PATH_AFTER
#     model_after=keras.models.load_model(SAVE_MODEL_PATH_AFTER)

#     model_before_pred=model_before.predict(inputs_before_test)
#     model_pred=model.predict(inputs_test)
#     model_after_pred=model_after.predict(inputs_after_test)

#     ac_before=r2(model_before_pred,targets_before_test)
#     ac_after=r2(model_after_pred, targets_after_test)
#     ac_main=r2(model_pred, targets_test)

#     print(ac_before, ac_main, ac_after, sep="\n")

    # #train the model
    # model.fit(inputs_train, targets_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # model_before.fit(inputs_before_train, targets_before_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # model_after.fit(inputs_after_train, targets_after_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # #save the model
    # model.save(SAVE_MODEL_PATH_MAIN)
    # model.save(SAVE_MODEL_PATH_BEFORE)
    # model.save(SAVE_MODEL_PATH_AFTER)

def test(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):

    model_path=SAVE_MODEL_PATH_MAIN
    model=keras.models.load_model(SAVE_MODEL_PATH_MAIN)

    

    inputs_test=INPUTS_TEST
    targets_test=TARGETS_TEST
    #print(len(inputs_test))
    #print(inputs_test)
    #print(targets_test)
    model_pred=model.predict(inputs_test)

    #targets_test ko one-hot
    targets_test=keras.utils.to_categorical(targets_test, num_classes=38)
    #print(targets_test)

    #target_test=TARGET_TEST
    r2indi=r2(targets_test,model_pred)

    print(r2indi)

def test_A(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):

    model_path=SAVE_MODEL_PATH_MAIN
    model=keras.models.load_model(SAVE_MODEL_PATH_AFTER)

    

    inputs_test=INPUTS_A_TEST
    targets_test=TARGETS_A_TEST
    #print(len(inputs_test))
    #print(inputs_test)
    #print(targets_test)
    model_pred=model.predict(inputs_test)

    #targets_test ko one-hot
    targets_test=keras.utils.to_categorical(targets_test, num_classes=38)
    #print(targets_test)

    #target_test=TARGET_TEST
    r2indi=r2(targets_test,model_pred)

    print(r2indi)

def test_B(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):

    model_path=SAVE_MODEL_PATH_MAIN
    model=keras.models.load_model(SAVE_MODEL_PATH_AFTER)

    

    inputs_test=INPUTS_B_TEST
    targets_test=TARGETS_B_TEST
    #print(len(inputs_test))
    #print(inputs_test)
    #print(targets_test)
    model_pred=model.predict(inputs_test)

    #targets_test ko one-hot
    targets_test=keras.utils.to_categorical(targets_test, num_classes=38)
    #print(targets_test)

    #target_test=TARGET_TEST
    r2indi=r2(targets_test,model_pred)

    print(r2indi)

#----------------------------

def all_test(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE, temperature=1):

    model_path=SAVE_MODEL_PATH_MAIN
    model_main=keras.models.load_model(SAVE_MODEL_PATH_MAIN)
    model_after=keras.models.load_model(SAVE_MODEL_PATH_AFTER)
    model_before=keras.models.load_model(SAVE_MODEL_PATH_BEFORE)
    

    inputs_B_test=INPUTS_B_TEST
    targets_B_test=TARGETS_B_TEST
    inputs_A_test=INPUTS_A_TEST
    targets_A_test=TARGETS_A_TEST
    inputs_test=INPUTS_TEST
    targets_test=TARGETS_TEST

    #print(len(inputs_test))
    #print(inputs_test)
    # #print(targets_test)
    # model_before_pred=model_before.predict(INPUTS_B_TEST)
    # model_after_pred=model_after.predict(INPUTS_A_TEST)
    # model_pred=model_main.predict(INPUTS_TEST)

    probab=model_main.predict(inputs_test)
    probab_A=model_after.predict(inputs_A_test)
    probab_B=model_before.predict(inputs_B_test)

    result=[]
    for i in range(len(inputs_test)):
        

        prob=probab[i]
        prob_A=probab_A[i]
        prob_B=probab_B[i]

        outint=_sample_with_temperature(prob,temperature)
        outint_A=_sample_with_temperature(prob_A,temperature)
        outint_B=_sample_with_temperature(prob_B,temperature)

        #choose something from the three options
        if outint_B==0 or outint_A==37:
            result.append(outint)
        elif outint_A-1==outint_B+1:
            result.append(outint_A-1)
        else:
            result.append(outint)

    #print(len(result))
    #print(len(targets_test))
    print(np.square(np.subtract(targets_test,result).mean()))
    #targets_test ko one-hot
    #targets_test=keras.utils.to_categorical(TARGETS_TEST, num_classes=38)
    #print(targets_test)

    #target_test=TARGET_TEST
    #r2indi=r2(targets_test,model_pred)

    #print(r2indi)


if __name__=="__main__":
    inputs_B_train, targets_B_train, inputs_B_test, targets_B_test, inputs_train, targets_train, inputs_test, targets_test, inputs_A_train, targets_A_train, inputs_A_test, targets_A_test=generating_training_sequences(38)

    INPUTS_TRAIN=inputs_train
    TARGETS_TRAIN=targets_train
    INPUTS_TEST=inputs_test
    TARGETS_TEST=targets_test
    INPUTS_A_TRAIN=inputs_A_train
    INPUTS_A_TEST=inputs_A_test
    INPUTS_B_TRAIN=inputs_B_train
    INPUTS_B_TEST=inputs_B_test
    TARGETS_A_TRAIN=targets_A_train
    TARGETS_A_TEST=targets_A_test
    TARGETS_B_TRAIN=targets_B_train
    TARGETS_B_TEST=targets_B_test

    train()
    train_A()
    train_B()
    test()
    test_A()
    test_B()
    all_test()