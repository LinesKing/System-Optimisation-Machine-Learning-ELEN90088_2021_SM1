	)\???(??)\???(??!)\???(??	`Q?(X,@`Q?(X,@!`Q?(X,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$)\???(??;?O??n??A)\???(??Y?A`??"??*	     ?_@2U
Iterator::Model::ParallelMapV2
ףp=
??!?m۶m?A@)
ףp=
??1?m۶m?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!??????:@)?? ?rh??1??????:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???Q???!?<??<?7@);?O??n??1$I?$I?,@:Preprocessing2F
Iterator::Model?v??/??!?y??y?F@)?~j?t???1?0?0#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t???!?0?0#@)?~j?t???1?0?0#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Q?????!r?q?;@){?G?zt?1??????@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t19.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9`Q?(X,@I????t}U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;?O??n??;?O??n??!;?O??n??      ??!       "      ??!       *      ??!       2	)\???(??)\???(??!)\???(??:      ??!       B      ??!       J	?A`??"???A`??"??!?A`??"??R      ??!       Z	?A`??"???A`??"??!?A`??"??b      ??!       JCPU_ONLYY`Q?(X,@b q????t}U@