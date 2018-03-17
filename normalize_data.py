#! /usr/bin/python3

import sys, csv
import time
import os

tank1 = "0_"
tank2 = "1_"
tank3 = "2_"
power1 = "3_0"
power2 = "3_1"
power3 = "3_2"
power4 = "3_3"
power5 = "3_4"
power6 = "3_5"
OXIGEN = str(0)
NITROGEN = str(1)
SST = str(2)
AMMONIA = str(3)
VALVE = str(4)
FLOW = str(5)

def write_date(date):
    return time.strftime("%d/%m/%Y %H:%M:%S", date)

def get_date(item):
    try:
        retval = time.strptime(item, "%d/%m/%Y %H:%M:%S")
    except ValueError:
        retval = time.strptime(item, "%d/%m/%Y %H:%M")

    return retval

def parse_code(code):
     # FLOW
    if code == "SO01_01-aMisura":
        code = tank1 + FLOW
    elif code == "SO01_02-aMisura":
        code = tank2 + FLOW
    elif code == "SO01_03-aMisura":
        code = tank3 + FLOW
    # SST
    elif code == "SO01_04-aMisura":
        code = tank1 + SST
    elif code == "SO01_05-aMisura":
        code = tank2 + SST
    elif code == "SO01_06-aMisura":
        code = tank3 + SST
    # OXIGEN
    elif code == "SO03_01-aMisura":
        code = tank1 + OXIGEN
    elif code == "SO05_01-aMisura":
        code = tank2 + OXIGEN
    elif code == "SO07_01-aMisura":
        code = tank3 + OXIGEN
    # AMMONIA
    elif code == "SO03_04-aMisura":
        code = tank1 + AMMONIA
    elif code == "SO05_04-aMisura":
        code = tank2 + AMMONIA
    elif code == "SO07_04-aMisura":
        code = tank3 + AMMONIA
    # NITROGEN
    elif code == "SO03_05-aMisura":
        code = tank1 + NITROGEN
    elif code == "SO05_05-aMisura":
        code = tank2 + NITROGEN
    elif code == "SO07_05-aMisura":
        code = tank3 + NITROGEN
    # VALVE
    elif code == "SO03_07-aMisura":
        code = tank1 + VALVE
    elif code == "SO05_07-aMisura":
        code = tank2 + VALVE
    elif code == "SO07_07-aMisura":
        code = tank3 + VALVE
    # POWER
    elif code == "CR03_01-aAssorbimento":
        code = power1
    elif code == "CR03_02-aAssorbimento":
        code = power2
    elif code == "CR03_03-aAssorbimento":
        code = power3
    elif code == "CR03_04-aAssorbimento":
        code = power4
    elif code == "CR03_05-aAssorbimento":
        code = power5
    elif code == "CR03_06-aAssorbimento":
        code = power6
    else:
        print("unknown code: " + code)
        exit(1)
    return code

def normalize_file(input_name, output_name):
    reader = csv.reader(open(input_name, 'r'), delimiter=";")

    sortedlist = sorted(reader, key=lambda line : get_date(line[0]), reverse=False)

    with open(output_name, 'w') as out:
        for line  in sortedlist:
            if len(line) == 3:
                date, code, measure = line
                date = get_date(date)
            elif len(line) > 3:
                date, code, measure = line[0:3]
                date = get_date(date)
                if line[3] != '':
                    print("extra items: " + str(line))
            else:
                print("missing items: " + str(line))
                exit(1)
            if measure == "NaN" or measure == "nan":
                measure = -1
            code = parse_code(code)
            out.write("{};{};{}\n".format(write_date(date), code, measure))
        print('.', end='', flush=True)


if len(sys.argv) != 3:
    print("usage: {} input_root_directory output_root_directory".format(sys.argv[0]))
    exit(1)
else:
    dst_root_dir = sys.argv[2]
    if dst_root_dir[len(dst_root_dir) -1] != '/':
        dst_root_dir = dst_root_dir + '/'

    src_root_dir = sys.argv[1]
    if src_root_dir[len(src_root_dir) -1] != '/':
        src_root_dir = src_root_dir + '/'

    for dir_it in os.listdir(src_root_dir):
        dst_dir = dst_root_dir + dir_it[13:] + "/"
        src_dir = src_root_dir + dir_it + "/"
        print("\n{} to {}: ".format(src_dir, dst_dir), end='')

        # create destination directory if required.
        if not os.path.exists(os.path.dirname(dst_dir)):
            try:
                os.makedirs(os.path.dirname(dst_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        for src_file in os.listdir(src_dir):
            src_file_name = src_dir + src_file
            dst_file_name = dst_dir + src_file[6:]
            normalize_file(src_file_name, dst_file_name)
