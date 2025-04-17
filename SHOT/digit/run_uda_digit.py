import subprocess
num_runs = 5  # Number of batches
batch_size = 20  # Number of parallel processes per batch
for i in range (1,10):
    for batch in range(1, num_runs + 1):  
        processes = []

        for j in range(1, batch_size + 1): 
    # want to run each 50 times 
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.1 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.2 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.3 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.4 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.5 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.6 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.7 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.8 --output ckps_digits
    # python uda_digit.py --dset mnistelection --cls_par 0.1 --dset_size 0.9 --output ckps_digits
    # Run from 51 to 100
            command = ["python", "uda_digit.py", "--dset_size",str(i / 10), "--cls_par","0.1","--output","ckps_digits","--iteration",str((j * batch)),"--dset","mnistelection"]
            p = subprocess.Popen(command)
            processes.append(p)
        for p in processes:
            p.wait()

        print(f"Batch {batch} completed ({batch_size * batch} total runs so far) On set size {i}.")

print("All processes have completed.")