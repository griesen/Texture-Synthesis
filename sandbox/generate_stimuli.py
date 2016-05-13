import os
import sys
from subprocess import Popen, PIPE

def get_input_images(path):
    input_ims = []
    for filename in os.listdir(path):
        if '.jpg' in filename:
            input_ims.append(filename)
    return input_ims


def process_images(path, input_ims, out_path):
    with open('job_numbers.txt', 'w') as job_file:
        for im_file in input_ims:
            im_name = im_file[:im_file.find('.')]
            with open('gen_stim.sbatch') as infile:
                lines = infile.readlines()
            outfile_name = 'gen_stim_'+im_name+'.sbatch'
            with open(outfile_name, 'w') as outfile:
                for line in lines:
                    if '--job-name' in line:
                        line = line[:line.find('=')+1] + 'gen_stim_' + im_name + '\n'
                    if '--output' in line:
                        line = line[:line.find('=')+1] + 'gen_stim_' + im_name + '.out' + '\n'
                    if '--error' in line:
                        line = line[:line.find('=')+1] + 'gen_stim_' + im_name + '.err' + '\n'
                    outfile.write(line)
                outfile.write(' '.join(['python', 'gen_stim_layers.py', path, im_file, out_path]))

            ps = Popen(('sbatch', outfile_name), stdout=PIPE)
            output = ps.communicate()[0]
            os.remove(outfile_name)
            job_file.write(output)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        assert os.path.isdir(path), "Not a valid path files"
    else:
        path = '.'
    if len(sys.argv) == 3:
        out_path = sys.argv[2]
    else:
        out_path = os.path.join(path,'generated')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    input_ims = get_input_images(path)
    assert len(input_ims), "No jpg images in the directory"
    process_images(path, input_ims, out_path)