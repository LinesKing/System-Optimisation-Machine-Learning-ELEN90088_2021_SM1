	R???Q??R???Q??!R???Q??	g??)??5@g??)??5@!g??)??5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$R???Q??bX9?ȶ?A?????K??YNbX9???*	     Ѐ@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Zd;???!????)$G@)?????K??1>?،?@@:Preprocessing2U
Iterator::Model::ParallelMapV2P??n???!?? ;J?7@)P??n???1?? ;J?7@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat????Mb??!27^Ѵ?7@)??K7?A??1?z?g?7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenatey?&1???!Lp27^?$@)y?&1???1Lp27^?$@:Preprocessing2F
Iterator::Model7?A`????!d3r??R;@);?O??n??1??k?
@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!ٌ??T@)9??v????1ٌ??T@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?I+???!???O\[ @)?I+???1???O\[ @:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????MbP?!27^Ѵ???)????MbP?127^Ѵ???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice????MbP?!27^Ѵ???)????MbP?127^Ѵ???:Preprocessing2a
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 21.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t10.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9g??)??5@I???5K?S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	bX9?ȶ?bX9?ȶ?!bX9?ȶ?      ??!       "      ??!       *      ??!       2	?????K???????K??!?????K??:      ??!       B      ??!       J	NbX9???NbX9???!NbX9???R      ??!       Z	NbX9???NbX9???!NbX9???b      ??!       JCPU_ONLYYg??)??5@b q???5K?S@