	ˡE?????ˡE?????!ˡE?????	?S???5#@?S???5#@!?S???5#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ˡE?????????????AR???Q??Y?&1???*	     ?[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??!]t?E]@@)???Q???1E]t?E;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Q?????!?E]t??@)y?&1???1t?E]t9@:Preprocessing2U
Iterator::Model::ParallelMapV2/?$???!]t?E3@)/?$???1]t?E3@:Preprocessing2F
Iterator::ModelV-???!\t?E]:@)????Mb??1]t?E@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!t?E]t@)y?&1?|?1t?E]t@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#??~j???!袋.?hR@)?~j?t?x?1?E]t?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?x?!?E]t?@)?~j?t?x?1?E]t?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+??????!?袋.?A@)????Mbp?1]t?E@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t18.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?S???5#@I?5K?O?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "      ??!       *      ??!       2	R???Q??R???Q??!R???Q??:      ??!       B      ??!       J	?&1????&1???!?&1???R      ??!       Z	?&1????&1???!?&1???b      ??!       JCPU_ONLYY?S???5#@b q?5K?O?V@