target="$1"
cmd="$2"
make_target=""

if [ -z "$cmd" ]; then
    cmd="build"
fi

if [ "$cmd" == "clean" ]; then
    make_target="clean"
fi

export LANG=en_US.US-ASCII
export PATH="/Applications/Xcode 8.2(B).app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin:/Applications/Xcode 8.2(B).app/Contents/Developer/usr/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if [ -z "$make_target" ]; then
    make -f tools/Makefile.ios.armv7
    make -f tools/Makefile.ios.arm64
else
    make -f tools/Makefile.ios.armv7 "${make_target}"
    make -f tools/Makefile.ios.arm64 "${make_target}"
fi

if [ $cmd == "build" ]; then
    echo "Creating multi-arch static library ..."
    lipo -create build/armv7/${target} build/arm64/${target} -o build/${target}
fi

echo "All done"
