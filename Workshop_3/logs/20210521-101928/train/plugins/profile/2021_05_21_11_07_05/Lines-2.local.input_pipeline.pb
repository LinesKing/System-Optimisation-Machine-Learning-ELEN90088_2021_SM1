	+?????+?????!+?????	!?!? @!?!? @!!?!? @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+???????|?5^??AV-???Y9??v????*	     @X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatˡE?????!Ӱ?,O"E@)y?&1???1,O"Ӱ?<@:Preprocessing2U
Iterator::Model::ParallelMapV2?I+???!?Q?/?6@)?I+???1?Q?/?6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9??v????!?$2??*@)9??v????1?$2??*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;?O??n??!|q???2@);?O??n??1|q???"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;?O??n??!|q???"@);?O??n??1|q???"@:Preprocessing2F
Iterator::ModelV-???!Jd???=@)y?&1?|?1,O"Ӱ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?? ?rh??!?fy??Q@){?G?zt?1D?a?Y?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?I+???!?Q?/?6@)????Mbp?1?Q?/?~@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t16.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9!?!? @I???[?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??|?5^????|?5^??!??|?5^??      ??!       "      ??!       *      ??!       2	V-???V-???!V-???:      ??!       B      ??!       J	9??v????9??v????!9??v????R      ??!       Z	9??v????9??v????!9??v????b      ??!       JCPU_ONLYY!?!? @b q???[?V@