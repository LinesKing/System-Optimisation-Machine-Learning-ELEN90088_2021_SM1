	d;?O????d;?O????!d;?O????	Nd6?5@Nd6?5@!Nd6?5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$d;?O????;?O??n??A
ףp=
??Y?O??n??*	     ??@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapV-???!??!?g?D@)/?$????1E]t?E;@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatX9??v??!6?d?M65@)?ʡE????1? 8?	?4@:Preprocessing2U
Iterator::Model::ParallelMapV2J+???!xI??x1@)J+???1xI??x1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate?? ?rh??!>???>(@)???x?&??1??????'@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapZd;?O???!f 	?7f@@)?p=
ף??1,?Z?C,'@:Preprocessing2F
Iterator::Model?Zd;??!????3?5@)?~j?t???1?b??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!______@)?I+???1______@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?I+???!______??)?I+???1______??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mb`?!?.?????)????Mb`?1?.?????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor????MbP?!?.?????)????MbP?1?.?????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice????MbP?!?.?????)????MbP?1?.?????:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 21.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t10.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Nd6?5@I??f?J?S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;?O??n??;?O??n??!;?O??n??      ??!       "      ??!       *      ??!       2	
ףp=
??
ףp=
??!
ףp=
??:      ??!       B      ??!       J	?O??n???O??n??!?O??n??R      ??!       Z	?O??n???O??n??!?O??n??b      ??!       JCPU_ONLYYNd6?5@b q??f?J?S@