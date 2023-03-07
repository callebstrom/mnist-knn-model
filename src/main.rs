use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use bincode::serialize;
use idx_parser::matrix::Matrix;
use idx_parser::IDXFile;
use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

use smartcore::math::distance::euclidian::Euclidian;
use smartcore::math::distance::hamming::Hamming;
use smartcore::math::distance::manhattan::Manhattan;
use smartcore::math::distance::Distances;
use smartcore::math::num::RealNumber;
use smartcore::metrics::accuracy;
use smartcore::neighbors::{
    knn_classifier::{KNNClassifier, KNNClassifierParameters},
    KNNWeightFunction,
};

const MAX_TRAINING_SAMPLES: usize = 100;

fn main() {
    let args: Vec<String> = env::args().collect();

    let operation = args[1].as_str();

    let (images, labels) = match operation {
        "train" => get_data(Dataset::Training),
        "validate" => get_data(Dataset::Validation),
        _ => panic!("Unknown operation"),
    };

    println!("dataset sample size {:#?}", images[0].len());
    println!("dataset labels size {:#?}", labels.len());

    match operation {
        "train" => {
            let model = train(&images, &labels);
            save_model("mnist.model", &model);
            println!("Saved model to mnist.model");
        }
        "validate" => {
            let model = load_model();
            let accuracy = validate(&model, &images, &labels);
            println!("Accuracy: {:#?}", accuracy);
        }
        _ => panic!("Unknown operation"),
    };
}

enum Dataset {
    Training,
    Validation,
}

fn get_data(dataset: Dataset) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (images_path, labels_path) = match dataset {
        Dataset::Training => (
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte",
        ),
        Dataset::Validation => (
            "data/validation-images.idx3-ubyte",
            "data/validation-labels.idx1-ubyte",
        ),
    };

    let (training_image_dimensions, training_images) = read_idx(Path::new(images_path)).unwrap();
    let (training_label_dimensions, training_labels) = read_idx(Path::new(labels_path)).unwrap();

    println!("Training images length {:#?}", training_images.len());
    println!("Training image dimensions {:#?}", training_image_dimensions);

    let training_labels_vec = training_labels
        .iter()
        .map(|row| flatten_row(&row))
        .take(MAX_TRAINING_SAMPLES)
        .collect::<Vec<Vec<f32>>>();

    let training_images_vec = training_images
        .iter()
        .map(|row| flatten_row(&row))
        .take(MAX_TRAINING_SAMPLES)
        .collect::<Vec<Vec<f32>>>();

    (training_images_vec, training_labels_vec)
}

fn flatten_row(matrix: &Box<Matrix>) -> Vec<f32> {
    match &**matrix {
        Matrix::Data(data) => match data {
            idx_parser::raw_data::RawData::UnsignedByte(b) => vec![*b as f32],
            _ => panic!("oh no"),
        },
        Matrix::Row(row) => row.iter().map(|mat| flatten_row(mat)).flatten().collect(),
    }
}

fn prepare_data<V: RealNumber>(
    samples: &Vec<Vec<V>>,
    labels: &Vec<Vec<V>>,
) -> (DenseMatrix<V>, Vec<V>) {
    let x = DenseMatrix::from_2d_vec(samples);
    let y = labels.iter().flatten().map(|u| *u).collect::<Vec<V>>();

    return (x, y);
}

fn train(
    training_samples: &Vec<Vec<f32>>,
    training_labels: &Vec<Vec<f32>>,
) -> KNNClassifier<f32, Euclidian> {
    let (x, y) = prepare_data(training_samples, training_labels);

    println!("Fitting classifier");

    let parameters: KNNClassifierParameters<f32, Euclidian> = KNNClassifierParameters::default()
        .with_distance(Distances::euclidian())
        .with_algorithm(KNNAlgorithmName::CoverTree)
        .with_weight(KNNWeightFunction::Distance)
        .with_k(3);

    KNNClassifier::fit(&x, &y, parameters).unwrap()
}

fn validate(
    model: &KNNClassifier<f32, Euclidian>, // &dyn Predictor<Vec<Vec<V>>, Vec<Vec<V>>>,
    validation_samples: &Vec<Vec<f32>>,    // &Vec<Vec<V>>,
    validation_labels: &Vec<Vec<f32>>,     //&Vec<Vec<V>>,
) -> f32 {
    let (x, y) = prepare_data(validation_samples, validation_labels);

    let y_hat = model.predict(&x).unwrap();
    accuracy(&y, &y_hat)
}

fn read_idx(path: &Path) -> Result<(Vec<u32>, Vec<Box<Matrix>>), ()> {
    let mut file = File::open(path).unwrap();

    let mut data = vec![];
    file.read_to_end(&mut data).unwrap();

    let idx = IDXFile::from_bytes(data).unwrap();

    Ok((idx.dimensions, idx.matrix_data))
}
fn load_model() -> KNNClassifier<f32, Euclidian> {
    let mut buf: Vec<u8> = Vec::new();
    File::open("mnist.model")
        .and_then(|mut f| f.read_to_end(&mut buf))
        .expect("Can not load model");
    bincode::deserialize(&buf).expect("Can not deserialize the model")
}

fn save_model(name: &str, model: &KNNClassifier<f32, Euclidian>) {
    let knn_bytes = serialize(model).expect("Can not serialize the model");
    File::create(name)
        .and_then(|mut f| f.write_all(&knn_bytes))
        .expect("Can not persist model");
    println!("Successfully saved model");
}
