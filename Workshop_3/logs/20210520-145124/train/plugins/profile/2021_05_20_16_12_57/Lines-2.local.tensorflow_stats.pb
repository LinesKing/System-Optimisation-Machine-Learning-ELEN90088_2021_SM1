"?j
BHostIDLE"IDLE1     ??@A     ??@a???????i????????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     @?@9     @?@A     @?@I     @?@aꚆz$???iI?Me????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      B@9      B@A      B@I      B@a??hDK?k?iꑰJ ???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?L@9     ?L@A      B@I      B@a??hDK?k?i?R??<???Unknown
iHostWriteSummary"WriteSummary(1      A@9      A@A      A@I      A@a&?b2.j?i??
1V???Unknown?
tHost_FusedMatMul"sequential_9/dense_29/Relu(1      =@9      =@A      =@I      =@a\oT?uTf?iX
??l???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      8@9      8@A      8@I      8@a??E??zb?i9Pf\ ???Unknown
?HostReadVariableOp"+sequential_9/dense_29/MatMul/ReadVariableOp(1      8@9      8@A      8@I      8@a??E??zb?i?>9{????Unknown
d	HostDataset"Iterator::Model(1     ?F@9     ?F@A      5@I      5@a?$==?+`?i??{??????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      4@9      4@A      4@I      4@aJvth??^?iz0????Unknown
VHostSum"Sum_2(1      0@9      0@A      0@I      0@ao+] ѣX?i<?_????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      .@9      .@A      .@I      .@a?XW?W?i?g???????Unknown
?HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      .@9      .@A      .@I      .@a?XW?W?ih?Ιx????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      *@9      *@A      *@I      *@aJ?K?T?iB??&{????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      (@9      (@A      (@I      (@a??E??zR?i2?/??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      (@9      (@A      (@I      (@a??E??zR?i"???????Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_30/MatMul(1      (@9      (@A      (@I      (@a??E??zR?i"r3????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      2@9      2@A      $@I      $@aJvth??N?i0?b?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aJvth??N?iN\?ԙ	???Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_31/MatMul(1      $@9      $@A      $@I      $@aJvth??N?ilyM???Unknown
[HostAddV2"Adam/add(1      "@9      "@A      "@I      "@a??hDK?K?i???;???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a??hDK?K?iԭ?+)???Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a??hDK?K?iȉ>&???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @ao+] ѣH?iS??2@,???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @ao+] ѣH?i??'i2???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @ao+] ѣH?i?b?8???Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_29/MatMul(1       @9       @A       @I       @ao+] ѣH?i4%??>???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @ao+] ѣH?i<??D???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?Q?V?E?i?P??GJ???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?Q?V?E?iCep??O???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_9/dense_31/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?Q?V?E?i?y/?U???Unknown
t Host_FusedMatMul"sequential_9/dense_30/Relu(1      @9      @A      @I      @a?Q?V?E?i??ZsZ???Unknown
w!Host_FusedMatMul"sequential_9/dense_31/BiasAdd(1      @9      @A      @I      @a?Q?V?E?ii??0?_???Unknown
v"HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a??E??zB?i????ud???Unknown
[#HostPow"
Adam/Pow_1(1      @9      @A      @I      @a??E??zB?iY??i???Unknown
z$HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??E??zB?i??OV?m???Unknown
~%HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??E??zB?iI??Rr???Unknown
?&HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??E??zB?i?????v???Unknown
?'HostMatMul",gradient_tape/sequential_9/dense_30/MatMul_1(1      @9      @A      @I      @a??E??zB?i9?{?{???Unknown
Y(HostPow"Adam/Pow(1      @9      @A      @I      @aJvth??>?i??i???Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aJvth??>?iW(L?B????Unknown
v*HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aJvth??>?i?6?E????Unknown
?+HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aJvth??>?iuE???????Unknown
?,HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @ao+] ѣ8?iQ?X
????Unknown
V-HostMean"Mean(1      @9      @A      @I      @ao+] ѣ8?i?\??????Unknown
?.HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @ao+] ѣ8?idhM3????Unknown
?/HostBiasAddGrad"7gradient_tape/sequential_9/dense_29/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ao+] ѣ8?i	t6?G????Unknown
?0HostBiasAddGrad"7gradient_tape/sequential_9/dense_30/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ao+] ѣ8?i?ZA\????Unknown
?1HostMatMul",gradient_tape/sequential_9/dense_31/MatMul_1(1      @9      @A      @I      @ao+] ѣ8?iS?~?p????Unknown
t2HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a??E??z2?i??????Unknown
]3HostCast"Adam/Cast_1(1      @9      @A      @I      @a??E??z2?i˜?r????Unknown
e4HostAddN"Adam/gradients/AddN(1      @9      @A      @I      @a??E??z2?i??O?^????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??E??z2?iC??)?????Unknown
\6HostGreater"Greater(1      @9      @A      @I      @a??E??z2?i?????????Unknown
?7HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      5@9      5@A      @I      @a??E??z2?i?? ?L????Unknown
?8HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??E??z2?iwȻ<?????Unknown
r9HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a??E??z2?i3?V??????Unknown
?:HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??E??z2?i????:????Unknown
v;HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??E??z2?i???O?????Unknown
v<HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??E??z2?ig?'?ٶ???Unknown
b=HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??E??z2?i#??)????Unknown
?>HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a??E??z2?i??]bx????Unknown
~?HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??E??z2?i???ǽ???Unknown
?@HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??E??z2?iW?????Unknown
?AHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a??E??z2?i/uf????Unknown
?BHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a??E??z2?i??е????Unknown
~CHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a??E??z2?i?(e,????Unknown
?DHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??E??z2?iG1 ?T????Unknown
?EHostReadVariableOp",sequential_9/dense_29/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??E??z2?i:???????Unknown
?FHostReadVariableOp",sequential_9/dense_31/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??E??z2?i?B6??????Unknown
rGHostSigmoid"sequential_9/dense_31/Sigmoid(1      @9      @A      @I      @a??E??z2?i{KњB????Unknown
~HHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @ao+] ѣ(?iNQ???????Unknown
vIHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @ao+] ѣ(?i!W?W????Unknown
oJHostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @ao+] ѣ(?i?\R?????Unknown
XKHostCast"Cast_3(1       @9       @A       @I       @ao+] ѣ(?i?b?k????Unknown
XLHostCast"Cast_5(1       @9       @A       @I       @ao+] ѣ(?i?h+??????Unknown
XMHostEqual"Equal(1       @9       @A       @I       @ao+] ѣ(?imn=	?????Unknown
aNHostIdentity"Identity(1       @9       @A       @I       @ao+] ѣ(?i@tOF
????Unknown?
TOHostMul"Mul(1       @9       @A       @I       @ao+] ѣ(?iza??????Unknown
vPHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @ao+] ѣ(?i?s?????Unknown
vQHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @ao+] ѣ(?i?????????Unknown
?RHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @ao+] ѣ(?i???:3????Unknown
}SHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @ao+] ѣ(?i_??w?????Unknown
`THostDivNoNan"
div_no_nan(1       @9       @A       @I       @ao+] ѣ(?i2???G????Unknown
uUHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @ao+] ѣ(?i????????Unknown
wVHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @ao+] ѣ(?iآ?.\????Unknown
xWHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @ao+] ѣ(?i???k?????Unknown
?XHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @ao+] ѣ(?i~??p????Unknown
?YHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @ao+] ѣ(?iQ???????Unknown
?ZHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @ao+] ѣ(?i$?'#?????Unknown
?[HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @ao+] ѣ(?i??9`????Unknown
?\HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @ao+] ѣ(?i??K??????Unknown
?]HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1       @9       @A       @I       @ao+] ѣ(?i??]?#????Unknown
?^Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @ao+] ѣ(?ip?o?????Unknown
?_HostReluGrad",gradient_tape/sequential_9/dense_29/ReluGrad(1       @9       @A       @I       @ao+] ѣ(?iCׁT8????Unknown
?`HostReluGrad",gradient_tape/sequential_9/dense_30/ReluGrad(1       @9       @A       @I       @ao+] ѣ(?iݓ??????Unknown
?aHostReadVariableOp"+sequential_9/dense_30/MatMul/ReadVariableOp(1       @9       @A       @I       @ao+] ѣ(?i????L????Unknown
?bHostReadVariableOp"+sequential_9/dense_31/MatMul/ReadVariableOp(1       @9       @A       @I       @ao+] ѣ(?i????????Unknown
tcHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??ao+] ѣ?i??@*?????Unknown
vdHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??ao+] ѣ?i???Ha????Unknown
veHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??ao+] ѣ?iw?Rg&????Unknown
VfHostCast"Cast(1      ??9      ??A      ??I      ??ao+] ѣ?i`?ۅ?????Unknown
jgHostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??ao+] ѣ?iI?d??????Unknown
whHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??ao+] ѣ?i2???u????Unknown
yiHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??ao+] ѣ?i?v?:????Unknown
?jHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??ao+] ѣ?i     ???Unknown
?kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??ao+] ѣ?iw?D?b ???Unknown
?lHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??ao+] ѣ?i??? ???Unknown
?mHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??ao+] ѣ?ia?ͭ'???Unknown
?nHostReadVariableOp",sequential_9/dense_30/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??ao+] ѣ?i?=????Unknown
+oHostCast"Cast_4(i?=????Unknown*?j
uHostFlushSummaryWriter"FlushSummaryWriter(1     @?@9     @?@A     @?@I     @?@a 4E?A??i 4E?A???Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      B@9      B@A      B@I      B@a?|~'???i(?????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?L@9     ?L@A      B@I      B@a?|~'???i6???????Unknown
iHostWriteSummary"WriteSummary(1      A@9      A@A      A@I      A@a??u??1??i͝?<????Unknown?
tHost_FusedMatMul"sequential_9/dense_29/Relu(1      =@9      =@A      =@I      =@ak??%????i???-B???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      8@9      8@A      8@I      8@a BST??i?\???????Unknown
?HostReadVariableOp"+sequential_9/dense_29/MatMul/ReadVariableOp(1      8@9      8@A      8@I      8@a BST??i??K?oS???Unknown
dHostDataset"Iterator::Model(1     ?F@9     ?F@A      5@I      5@a???.???i?=?C?????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      4@9      4@A      4@I      4@aVÊ??v??i?h̝?<???Unknown
V
HostSum"Sum_2(1      0@9      0@A      0@I      0@a?opxņ?i?$??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      .@9      .@A      .@I      .@a?h? Y??i??3R????Unknown
?HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      .@9      .@A      .@I      .@a?h? Y??i:eن?B???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      *@9      *@A      *@I      *@a+2Z?q???i?FN?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      (@9      (@A      (@I      (@a BST??i??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      (@9      (@A      (@I      (@a BST??ih? Y???Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_30/MatMul(1      (@9      (@A      (@I      (@a BST??i?:??Y???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      2@9      2@A      $@I      $@aVÊ??v|?i??S7?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aVÊ??v|?i)?l??????Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_31/MatMul(1      $@9      $@A      $@I      $@aVÊ??v|?i????r???Unknown
[HostAddV2"Adam/add(1      "@9      "@A      "@I      "@a?|~'?y?iv????7???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a?|~'?y?i<?/?j???Unknown?
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a?|~'?y?i?|~'????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?opx?v?i?]o?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a?opx?v?i?>`=????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?opx?v?i}Q?&???Unknown
~HostMatMul"*gradient_tape/sequential_9/dense_29/MatMul(1       @9       @A       @I       @a?opx?v?i[ BST???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?opx?v?i9?2ށ???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aV"ab??s?i`??ŷ????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aV"ab??s?i??jX?????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_9/dense_31/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aV"ab??s?i?/?j????Unknown
tHost_FusedMatMul"sequential_9/dense_30/Relu(1      @9      @A      @I      @aV"ab??s?i/B?}D!???Unknown
w Host_FusedMatMul"sequential_9/dense_31/BiasAdd(1      @9      @A      @I      @aV"ab??s?it?I???Unknown
v!HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a BSTq?i??aEFk???Unknown
["HostPow"
Adam/Pow_1(1      @9      @A      @I      @a BSTq?i|Q
zn????Unknown
z#HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a BSTq?i ????????Unknown
~$HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a BSTq?i??[??????Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a BSTq?iE?????Unknown
?&HostMatMul",gradient_tape/sequential_9/dense_30/MatMul_1(1      @9      @A      @I      @a BSTq?i???L???Unknown
Y'HostPow"Adam/Pow(1      @9      @A      @I      @aVÊ??vl?iOv9#?2???Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aVÊ??vl?i???N???Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aVÊ??vl?iՋR?sk???Unknown
?*HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aVÊ??vl?i?ߦ?????Unknown
?+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?opx?f?i??O?????Unknown
V,HostMean"Mean(1      @9      @A      @I      @a?opx?f?i????u????Unknown
?-HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?opx?f?i?c0;????Unknown
?.HostBiasAddGrad"7gradient_tape/sequential_9/dense_29/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?opx?f?i?Ҡ? ????Unknown
?/HostBiasAddGrad"7gradient_tape/sequential_9/dense_30/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?opx?f?i?A?????Unknown
?0HostMatMul",gradient_tape/sequential_9/dense_31/MatMul_1(1      @9      @A      @I      @a?opx?f?i???y????Unknown
t1HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a BSTa?i?֓?!???Unknown
]2HostCast"Adam/Cast_1(1      @9      @A      @I      @a BSTa?i.W*??2???Unknown
e3HostAddN"Adam/gradients/AddN(1      @9      @A      @I      @a BSTa?ip?~??C???Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a BSTa?i?????T???Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a BSTa?i?P'??e???Unknown
?6HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      5@9      5@A      @I      @a BSTa?i6?{w???Unknown
?7HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a BSTa?ix??1????Unknown
r8HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a BSTa?i?J$L,????Unknown
?9HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a BSTa?i??xf@????Unknown
v:HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a BSTa?i>?̀T????Unknown
v;HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a BSTa?i?D!?h????Unknown
b<HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a BSTa?iu?|????Unknown
?=HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a BSTa?i??ϐ????Unknown
~>HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a BSTa?iF>??????Unknown
??HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a BSTa?i??r????Unknown
?@HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a BSTa?i????!???Unknown
?AHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a BSTa?i89?2???Unknown
~BHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a BSTa?iN?oS?C???Unknown
?CHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a BSTa?i???m	U???Unknown
?DHostReadVariableOp",sequential_9/dense_29/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a BSTa?i?1?f???Unknown
?EHostReadVariableOp",sequential_9/dense_31/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a BSTa?i?l?1w???Unknown
rFHostSigmoid"sequential_9/dense_31/Sigmoid(1      @9      @A      @I      @a BSTa?iV???E????Unknown
~GHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1       @9       @A       @I       @a?opx?V?i??x?????Unknown
vHHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @a?opx?V?iXG15????Unknown
oIHostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @a?opx?V?i?~i?m????Unknown
XJHostCast"Cast_3(1       @9       @A       @I       @a?opx?V?iZ???е???Unknown
XKHostCast"Cast_5(1       @9       @A       @I       @a?opx?V?i???i3????Unknown
XLHostEqual"Equal(1       @9       @A       @I       @a?opx?V?i\%&?????Unknown
aMHostIdentity"Identity(1       @9       @A       @I       @a?opx?V?i?\J??????Unknown?
TNHostMul"Mul(1       @9       @A       @I       @a?opx?V?i^???[????Unknown
vOHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?opx?V?i?˺Z?????Unknown
vPHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?opx?V?i`?!????Unknown
?QHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?opx?V?i?:+Ӄ???Unknown
}RHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?opx?V?ibrc?????Unknown
`SHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?opx?V?i㩛KI???Unknown
uTHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?opx?V?id???'???Unknown
wUHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?opx?V?i??3???Unknown
xVHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?opx?V?ifPD?q>???Unknown
?WHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?opx?V?i??|<?I???Unknown
?XHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a?opx?V?ih???6U???Unknown
?YHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?opx?V?i??촙`???Unknown
?ZHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?opx?V?ij.%q?k???Unknown
?[HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a?opx?V?i?e]-_w???Unknown
?\HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1       @9       @A       @I       @a?opx?V?il????????Unknown
?]Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a?opx?V?i??ͥ$????Unknown
?^HostReluGrad",gradient_tape/sequential_9/dense_29/ReluGrad(1       @9       @A       @I       @a?opx?V?inb?????Unknown
?_HostReluGrad",gradient_tape/sequential_9/dense_30/ReluGrad(1       @9       @A       @I       @a?opx?V?i?C>?????Unknown
?`HostReadVariableOp"+sequential_9/dense_30/MatMul/ReadVariableOp(1       @9       @A       @I       @a?opx?V?ip{v?L????Unknown
?aHostReadVariableOp"+sequential_9/dense_31/MatMul/ReadVariableOp(1       @9       @A       @I       @a?opx?V?i񲮖?????Unknown
tbHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a?opx?F?i????`????Unknown
vcHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?opx?F?is??R????Unknown
vdHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?opx?F?i4??????Unknown
VeHostCast"Cast(1      ??9      ??A      ??I      ??a?opx?F?i?!u????Unknown
jfHostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a?opx?F?i?=;m&????Unknown
wgHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?opx?F?iwYW??????Unknown
yhHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?opx?F?i8us)?????Unknown
?iHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?opx?F?i????:????Unknown
?jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?opx?F?i?????????Unknown
?kHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?opx?F?i{??C?????Unknown
?lHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?opx?F?i<???N????Unknown
?mHostReadVariableOp",sequential_9/dense_30/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?opx?F?i?????????Unknown
+nHostCast"Cast_4(i?????????Unknown2CPU