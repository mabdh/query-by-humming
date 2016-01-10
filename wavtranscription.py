import os

# Download waon (http://sourceforge.net/project/showfiles.php?group_id=177685)
# Copy executable to project folder
# - dependencies
# - fftw (http://www.fftw.org/download.html)
# - libsnd (http://www.mega-nerd.com/libsndfile/#Download)
# - To build, 1. ./configure
#             2. make
#             3. sudo make install

def convert_wav_to_midi(filename):
    # filename without extension
    filewav = filename + ".wav"
    filemidi = filename + ".mid"
    os.system("./waon -i " + filewav + " -o " + filemidi + " -w 3 -n 4096 -s 2048")

def main():
    convert_wav_to_midi("classic")
    
main()