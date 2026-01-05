if [ -z "$1" ]; then
    echo "Usage: $0 <protein_name>"
    exit 1
fi

name=$1
wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip
mkdir -p ${name} # Use -p to avoid error if exists
unzip ${name}_protein.zip -d ${name}
rm ${name}_protein.zip