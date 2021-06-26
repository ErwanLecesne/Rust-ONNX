extern crate ndarray;
use ndarray::{Array,Array3,Array4, Axis};
use std::time::SystemTime;
use nifti::{NiftiObject, ReaderOptions, writer, InMemNiftiObject, NiftiVolume,IntoNdArray};
//use tract_ndarray::Array;
use tract_onnx::prelude::*;

fn main() {


    let start = SystemTime::now();


    let reader = ReaderOptions::new();

    let image = reader.read_file("/home/erwan/Téléchargements/acdc128.nii").unwrap();
    //use obj
    //let header = image.header();
    let mut volume = image.into_volume().into_ndarray::<f32>().unwrap();
    let dims = volume.shape();
    for v in dims.iter() {
        println!("input shape {:?}",v);
    }
    

    let mut niiwriter = writer::WriterOptions::new("/home/erwan/Téléchargements/input.nii");
    let mut write = niiwriter.write_nifti(&volume);
    


    volume.swap_axes(0, 3); 
    volume.swap_axes(1, 2); 
    volume.swap_axes(0, 1); //(B,C,H,W)
    //volume = volume.insert_axis(Axis(0));
    let mean = volume.sum() / volume.len() as f32;
    let std = volume.std(0.);

    volume = (volume - mean) / std; //Znormalization

  
 

    let model = tract_onnx::onnx()
    // load the model
    .model_for_path("/home/erwan/Téléchargements/dcunet_soft.onnx").unwrap()
    // specify input type and shape
    .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(16, 1, 128, 128))).unwrap()
    // optimize the model
    .into_optimized().unwrap()
    // make the model runnable and fix its inputs and outputs
    .into_runnable().unwrap();



    

    //let vol = Array4::<f32>::ones((16, 1, 128, 128));
    let image: Tensor = volume.into(); //

    let result = model.run(tvec!(image)).unwrap(); // make the inference 
    
    let prediction = (*result[0]).clone().into_array::<i64>().unwrap();
    let mut casted = prediction.map(|&e| e as f32);


    let dims_res = prediction.shape();

    
    for v in dims_res.iter() {
        println!("output shape {:?}",v);
    }


    casted.swap_axes(0, 2);

    

    let dims_pred= casted.shape();

    for v in dims_pred.iter() {
        println!("prediction shape  {:?}",v);
    }
    
    niiwriter = writer::WriterOptions::new("/Users/Erwan/Downloads/output.nii");
    write = niiwriter.write_nifti(&casted);

    let end = SystemTime::now();
    let difference = end.duration_since(start);
    println!("time to make it {:?}",difference);  
    
   
}